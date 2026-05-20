import os
import ray
import argparse
import socket
from omegaconf import OmegaConf
import torch
import hydra

from src.trainer.ray_trainer import RayPPOTrainer
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.fs import copy_to_local
from verl.trainer.ppo.reward import load_reward_manager

from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.utils.import_utils import load_extern_type
from verl.utils.device import is_cuda_available

def create_rl_dataset(data_paths, data_config, tokenizer, processor, map_index_list, is_train=True, data_slice=None):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    # from verl.utils.dataset.rl_dataset import RLHFDataset
    # from src.utils.dataset.rl_dataset import CustomizeRLHFDataset
    from src.utils.dataset.seed_dataset import SeedDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        # dataset_cls = RLHFDataset
        dataset_cls = SeedDataset   #CustomizeRLHFDataset
        if data_slice is None:      # if data_slice is not specified, create parquet file with all data size
            data_slice = len(map_index_list)

    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        map_index_list=map_index_list,
        customized_prompt_set=data_config.get("customized_prompt_set", None),
        data_slice = data_slice,
    )

    return dataset


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

        if (
                is_cuda_available
                and config.trainer.get("profile_steps") is not None
                and len(config.trainer.get("profile_steps", [])) > 0
        ):
            nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))

        timeline_json_file = config.ray_init.get("timeline_json_file", None)
        if timeline_json_file:
            ray.timeline(filename=timeline_json_file)

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        config.custom_reward_function

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from src.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            from verl.workers.fsdp_workers import CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.reward_manager in ['mdp']:
            if config.reward_model.reward_manager == 'mdp':
                from src.workers.reward_manager import MDPRewardManager
                val_reward_fn = MDPRewardManager(tokenizer, None, compute_score=None, reward_fn_key="data_source")
                reward_fn = MDPRewardManager(tokenizer, None, compute_score=None, reward_fn_key="data_source")
        else:
            reward_fn = load_reward_manager(
                config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
            )
            val_reward_fn = load_reward_manager(
                config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
            )


        # Add a reference policy worker if KL loss or KL reward is used.
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        if config.actor_rollout_ref.rollout.multi_turn.envs.env_config.env_name=="ChessPuzzles":
            from tasks.classic_games.chess.chess_puzzle_env import ChessPuzzleEnv_Wrapper
            _test_env = ChessPuzzleEnv_Wrapper(config.actor_rollout_ref.rollout.multi_turn.envs.env_config)
        elif config.actor_rollout_ref.rollout.multi_turn.envs.env_config.env_name=="ScienceWorld":
            from tasks.classic_games.scienceworld.scienceworld_env import ScienceWorldEnv_Wrapper
            _test_env = ScienceWorldEnv_Wrapper(config.actor_rollout_ref.rollout.multi_turn.envs.env_config)
        else:
            raise ValueError(f"Invalid environment: {config.actor_rollout_ref.rollout.multi_turn.envs.env_config.env_name}")

        total_num_game = _test_env.num_game
        train_val_split = int(
            config.actor_rollout_ref.rollout.multi_turn.envs.env_config.rollout.train_val_ratio * total_num_game)
        val_map_index = list(range(0, train_val_split))
        train_map_index = list(range(train_val_split, total_num_game))  # first ratio is the val set
        print(f"\n\ntrain_map_index: {train_map_index}, val_map_index: {val_map_index}\n\n")

        # reconfigurate val batch size according to val data size, choose the smaller one
        config.data.val_batch_size=min(len(val_map_index), config.data.val_batch_size)
        config.data.train_batch_size=min(len(train_map_index), config.data.train_batch_size)

        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor,
                                        val_map_index,is_train=False)
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor,
                                        train_map_index,is_train=True,)

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()
        # trainer.global_steps = 1
        # eval_ret = trainer._validate()
        # print(f"eval ret = {eval_ret}")


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler




if __name__ == "__main__":
    main()





