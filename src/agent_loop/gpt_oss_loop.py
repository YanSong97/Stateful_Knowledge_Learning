
import asyncio
import json
import logging
import os
import re
from typing import Any
from uuid import uuid4
import importlib

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register, _DummyConfig, AsyncLLMServerManager
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from src.workers.rollout.vllm_rollout.mcp_search_client import MCPDirectSearchClient, DirectHTTPSearchClient, DirectModelAPIClient
from src.utils.dataset.rl_dataset import CustomizeRLHFDataset

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("oss_agent")
class OSS_AgentLoop(AgentLoopBase):
    """Agent loop with custom SearchClient for search API execution"""

    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level SearchAgentLoop initialization")

        # oss new chat template
        oss_chat_template_path = "resources.models.gpt_oss_20b.tokenizer.GPT_OSS_20B_TEMPLATE"
        module_path, class_name = oss_chat_template_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        new_chat_template = getattr(module, class_name)
        tokenizer.chat_template = new_chat_template

        # Initialize tokenizer and config
        cls.tokenizer = tokenizer
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls

        # Search client configuration
        search_config = config.actor_rollout_ref.rollout.multi_turn.get('search_config', {})
        cls.topk = search_config.get('topk', 3)

        tools_config_file = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_config = OmegaConf.load(tools_config_file)
        backend = tool_config.tools[0].backend
        if backend == "mcp":
            cls.search_client = MCPDirectSearchClient(top_k = cls.topk, config=tool_config.tools[0].config)
        elif backend == 'http':
            cls.search_client = DirectHTTPSearchClient(search_url=tool_config.tools[0].config.url, topk=cls.topk)
        else:
            raise NotImplementedError(f"Backend {backend} not supported.")

        cls.search_mode = config.actor_rollout_ref.rollout.multi_turn.search_config.search_mode
        assert cls.search_mode in ["sequential", "parallel"], print(f"{cls.search_mode} is not supported")

        cls.model_api_client = DirectModelAPIClient()

        # Query/answer detection patterns (configurable)
        query_start_tag = search_config.get('query_start_tag', '<search>')
        query_end_tag = search_config.get('query_end_tag', '</search>')
        answer_start_tag = search_config.get('answer_start_tag', '<answer>')
        answer_end_tag = search_config.get('answer_end_tag', '</answer>')

        cls.start_tool_marker = search_config.get("start_tool_marker", "")

        cls.query_pattern = re.compile(f'{re.escape(query_start_tag)}(.*?){re.escape(query_end_tag)}', re.DOTALL)
        cls.answer_pattern = re.compile(f'{re.escape(answer_start_tag)}(.*?){re.escape(answer_end_tag)}', re.DOTALL)

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

        cls.tool_masking = config.actor_rollout_ref.rollout.multi_turn.post_process.tool_masking

        cls.search_response_prefix = config.actor_rollout_ref.rollout.multi_turn.search_config.get("search_response_prefix", "tool")
        cls.intermediate_instruction = config.actor_rollout_ref.rollout.multi_turn.get("intermediate_instruction", False)

        cls.tools_lst = [
            {"type": "function", "name": "search", "description": "Get external knowledge via a search engine",
             "parameters": {
                 "type": "object",
                 "properties": {"query": {"type": "string"}, "top_k": {"type": "int"}},
                 "required": ["query"],
                },
             }
        ]
        cls.reasoning_effort = "low"
        assert cls.reasoning_effort in ['low', 'medium', 'high']

    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        if self.search_mode == 'sequential':
            return await self.run_sequential(messages, sampling_params)
        elif self.search_mode == 'parallel':
            return await self.run_parallel(messages, sampling_params)
        else:
            raise NotImplementedError

    @rollout_trace_op
    async def run_sequential(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        search_metrics = {}
        request_id = uuid4().hex

        for i in messages:
            if i['role'] == 'system':
                i['reasoning_effort'] = self.reasoning_effort

        raw_message = [i for i in messages]         # for vllm communication
        template_message = [i for i in messages]    # for template apply

        # Initial prompt encoding
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                template_message, add_generation_prompt=False, tokenize=True
            ),
        )
        initial_prompt_len = len(prompt_ids)
        response_mask = []

        user_turns, assistant_turns = 0, 0
        early_stop_flag = 0     # if not stop at agent response

        while True:
            # Generate response from LLM
            with simple_timer("generate_sequences", metrics):
                # response_ids = await self.server_manager.generate(
                #     request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                # )
                response_lst = await self.model_api_client.request([raw_message], self.reasoning_effort)      # vllm server receive raw message
                response_dict = response_lst[0]

            tmp_raw, tmp_template, tmp_tool_call_lst = response_dict['raw'], response_dict['template'], response_dict['tool_call_list']

            # response_ids = await self.loop.run_in_executor(
            #     None,
            #     lambda: self.tokenizer.apply_chat_template(
            #         tmp_template, add_generation_prompt=False, tokenize=True
            #     )
            # )
            # response_ids = response_ids[len(self.system_prompt):]
            # prompt_ids += response_ids
            # response_mask += [1] * len(response_ids)
            assistant_turns += 1

            raw_message.extend(tmp_raw)
            template_message.extend(tmp_template)

            # Check termination conditions
            if len(response_mask) >= self.response_length:
                break
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            if self.max_user_turns and user_turns >= self.max_user_turns:
                break

            # Extract search query
            search_query, call_id, call_fn_name = self._extract_query(tmp_tool_call_lst)
            if not search_query:
                break  # No search query found, end conversation

            # Call search API using our SearchClient
            with simple_timer("tool_calls", metrics):
                try:
                    search_results, _search_metrics = await self._search_async(search_query)
                    if not search_results:
                        break  # Search failed or returned nothing
                except Exception as e:
                    logger.exception(f"Search API error: {e}")
                    break

            # metrics.update(search_metrics)
            for k, v in _search_metrics.items():
                search_metrics[k] = search_metrics.get(k, 0) + v

            # Format search response
            search_response, meta_data = self._format_rag_response(search_results)

            # processing
            raw_message.append(
                {"type": "function_call_output", "call_id": call_id, "output": search_response}
            )

            search_response_message = [{"role": "tool", "name": call_fn_name, "content": search_response}]
            template_message.extend(search_response_message)

            # search_response_ids = await self.loop.run_in_executor(
            #     None,
            #     lambda: self.tokenizer.apply_chat_template(
            #         search_response_message, add_generation_prompt=False, tokenize=True
            #     )
            # )
            # search_response_ids = search_response_ids[len(self.system_prompt):]
            #
            # # Check if adding search response would exceed max length
            # if len(response_mask) + len(search_response_ids) >= self.response_length:
            #     early_stop_flag = 1
            #     break
            #
            # # Add search response to sequence
            # prompt_ids += search_response_ids
            # response_mask += [0] * len(search_response_ids)  # Tool response not trained
            user_turns += 1

            # Check tool usage, if run out of tool call, force agent to answer, else give intermediate guidance
            if assistant_turns + 1 >= self.max_assistant_turns:     # next completion is the finale
                force_answer_message = [{"role": "user", "content": "No available tools at the moment. Generate your final conclusion without tool usage."}]
                raw_message.extend(force_answer_message)
                template_message.extend(force_answer_message)

                # force_answer_ids = await self.loop.run_in_executor(
                #     None,
                #     lambda: self.tokenizer.apply_chat_template(
                #         force_answer_message, add_generation_prompt=True, tokenize=True
                #     ),
                # )
                # force_answer_ids = force_answer_ids[len(self.system_prompt):]
                # prompt_ids += force_answer_ids
                # response_mask += [0] * len(force_answer_ids)

            else:
                intermediate_message = [{"role": "user", "content": "The tools are still avaiable at the moment. You can either use it or generate your conclusion without tool usage"}]
                raw_message.extend(intermediate_message)
                template_message.extend(intermediate_message)

                # intermediate_ids = await self.loop.run_in_executor(
                #     None,
                #     lambda: self.tokenizer.apply_chat_template(
                #         intermediate_message, add_generation_prompt=True, tokenize=True
                #     ),
                # )
                # intermediate_ids = intermediate_ids[len(self.system_prompt):]
                # prompt_ids += intermediate_ids
                # response_mask += [0] * len(intermediate_ids)


        # print(f"message = {template_message}")

        ret = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                template_message, add_generation_prompt=False,
                tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True
            ),
        )

        prompt_ids = ret.input_ids
        response_mask = ret.assistant_masks

        response_ids = prompt_ids[initial_prompt_len:]
        response_mask = response_mask[initial_prompt_len:]
        prompt_ids = prompt_ids[:initial_prompt_len]


        # Prepare final output
        # response_ids = prompt_ids[-len(response_mask):]
        # prompt_ids = prompt_ids[:len(prompt_ids) - len(response_mask)]

        # if self.tool_masking:
        #     response_mask = self._compute_masking(response_ids, response_mask)
        # print(f"metrics = {metrics}")
        metrics['others'] = {
            "early_stop": len(response_ids) > self.response_length
        }


        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            num_turns=user_turns + assistant_turns + 1,
            metrics=metrics,
        )
        return output

    @rollout_trace_op
    async def run_parallel(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        search_metrics = {}
        request_id = uuid4().hex

        # Initial prompt encoding
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []

        user_turns, assistant_turns = 0, 0

        while True:
            # Generate response from LLM
            with simple_timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
                pass


        return


    def _extract_query(self, tool_call_list: list):
        """Extract search query from response text using patterns"""
        if len(tool_call_list) == 0:
            return "", None, None

        fn_name = tool_call_list[0]['name']
        fn_args = json.loads(tool_call_list[0]['arguments'])
        call_id = tool_call_list[0]['call_id']

        return fn_args['query'], call_id, fn_name

    def filter_messages(self, messages):
        new_messages = []
        for idx, mem in enumerate(messages):
            if mem['role'] in ['system', "user", "tool"]:
                new_messages.append(mem)
                continue
            if mem['role'] in ['thinker']:
                # whether to merge with assistant response
                if messages[idx + 1]['role'] in ["assistant"] and idx + 2 >= len(messages):
                    new_messages.append(
                        {"role": "assistant", "content": messages[idx + 1]['content'], "thinking": mem['content']}
                    )
                    break
                else:
                    new_messages.append(
                        {"role": "assistant", "content": mem['content']}
                    )

            if mem['role'] in ['assistant']:
                new_messages.append(mem)

        return new_messages



    async def _search_async(self, query: str) -> Any:
        """Call the search API asynchronously"""
        # Run the synchronous batch_search in a thread pool
        # results = await self.loop.run_in_executor(
            # None,
            # lambda: self.search_client.batch_search([query])
        # )
        # Format the results using the client's formatter
        # results = await self.search_client.batch_search([query])

        return await self.search_client.batch_search([query])

    # def _format_search_response(self, results: Any) -> str:
    #     """Format search results for the conversation"""
    #     # TODO: Customize this formatting as needed for your use case
    #     if not results or not results[0]:
    #         return "<information>\nNo results found.\n</information>"
    #     # For now, just dump the first result as JSON
    #     return f"<information>\n{json.dumps(results[0], ensure_ascii=False, indent=2)}\n</information>"

    def _format_rag_response(self, results: Any):
        """Format search results for the conversation"""
        # TODO: Customize this formatting as needed for your use case

        if not results or not results[0]:
            return "<information>\nNo results found.\n</information>", {}

        # print(f"length of results = {results}")
        # print(f"dict key = {results[0].keys()}")

        if isinstance(results[0], str):     # error message
            searched_results = f"<information>\n{json.dumps(results[0], ensure_ascii=False, indent=2)}\n</information>"
            meta_data = {}
        else:
            final_ret = results[0]['result']

            if isinstance(final_ret, str):      # error message
                searched_results = f"<information>\n{json.dumps(final_ret, ensure_ascii=False, indent=2)}\n</information>"
                meta_data = {}
                return searched_results, meta_data

            # For now, just dump the first result as JSON
            searched_results = f"<information>\n{json.dumps(final_ret['info_str'], ensure_ascii=False, indent=2)}\n</information>"
            meta_data = {
                "engines": final_ret["engines"],
                "scores": final_ret["scores"],
                "urls": final_ret["urls"],
                "search_time": results[0]['search_time'],
                "summary_time": results[0]['summarize_time'],
                "total_tool_engine_time": results[0]['total_time'],
            }

        return searched_results, meta_data

    # def _compute_masking(self, response_ids: list[int], response_mask: list[int]):
    #     if not self.tool_masking:
    #         return response_mask
    #
    #     # decode to search tag
    #     response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
    #
    #     # find tag
    #     start_positions = [m.start() for m in re.finditer(re.escape(self.start_state_marker), response_text)]
final_results = []


async def _async_main():
    import argparse
    from pathlib import Path
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config/oss_loop_trainer.yaml")
    # dataset-style args (align with evaluation/scripts/eval_async_search.py)
    parser.add_argument("--data_source", type=str, nargs="+", default=["nq_search"], help="name(s) under datasets/ to load (expects test.parquet or *.jsonl)")
    parser.add_argument("--tool_config_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--output", type=str, default="logs/eval/gpt_oss/logs.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="limit number of samples for a quick run")
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Ensure tool config path is set
    if not config.actor_rollout_ref.rollout.multi_turn.tool_config_path:
        repo_root = Path(__file__).resolve().parents[2]
        tool_cfg = repo_root / "src" / "config" / "tool_cfg" / "local_search_config.yaml"
        config.actor_rollout_ref.rollout.multi_turn.tool_config_path = str(tool_cfg)

    # Model/tokenizer path overrides
    if args.model_path is not None:
        config.actor_rollout_ref.model.path = args.model_path
    elif config.actor_rollout_ref.model.get("path", None) is None:
        raise ValueError("A local model path is required. Pass --model_path or set actor_rollout_ref.model.path in config.")

    # Tokenizer from model path (can override)
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = config.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(tokenizer_path), trust_remote_code=True)

    # Minimal sampling params
    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        "stop": list(config.actor_rollout_ref.rollout.get("stop_words", [])),
    }

    # Multi-turn and stopwords (follow eval settings)
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = args.max_turns
    config.actor_rollout_ref.rollout.multi_turn.max_user_turns = args.max_turns
    config.actor_rollout_ref.rollout.response_length=8192
    config.actor_rollout_ref.rollout.stop_words=["</search>", "</answer>"]

    # Tool config override
    if args.tool_config_path is not None:
        config.actor_rollout_ref.rollout.multi_turn.tool_config_path = args.tool_config_path

    # Ensure dataset returns raw chat messages
    if not hasattr(config, "data"):
        config["data"] = {}
    config.data.return_raw_chat = True

    # Instantiate agent loop instance INSIDE a running event loop
    server_manager = None  # AsyncLLMServerManager(config, server_handles=[])
    agent = OSS_AgentLoop(
        trainer_config=_DummyConfig(config=config),
        server_manager=server_manager,
        tokenizer=tokenizer,
    )

    # Discover data files under datasets/{data_source}
    parquet_files = []
    jsonl_files = []
    for ds in args.data_source:
        # if a direct file path is provided
        if os.path.isfile(ds):
            if ds.endswith(".parquet"):
                parquet_files.append(ds)
            elif ds.endswith(".jsonl"):
                jsonl_files.append(ds)
            continue

        base = os.path.join("datasets", ds)
        test_parquet = os.path.join(base, "test.parquet")
        if os.path.exists(test_parquet):
            parquet_files.append(test_parquet)
        else:
            # collect jsonl files recursively
            for root, _, files in os.walk(base):
                for f in files:
                    if f.endswith(".jsonl"):
                        jsonl_files.append(os.path.join(root, f))

    if not parquet_files and not jsonl_files:
        raise FileNotFoundError(f"No dataset files found for {args.data_source}. Expect test.parquet or *.jsonl under datasets/<name>.")

    # Build output file if requested
    output_fp = None
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_fp = open(args.output, "w", encoding="utf-8")

    def iter_dataset(file_list, is_parquet: bool):
        if not file_list:
            return None
        ds_class = CustomizeRLHFDataset if is_parquet else JsonRLHFDataset
        dataset = ds_class(
            data_files=file_list if len(file_list) > 1 else file_list[0],
            tokenizer=tokenizer,
            processor=None,
            config=config.data,
            customized_prompt_set="tasks.general_agents.NQ.prompts.SEARCH_CUSTOMIZE_SET",
        )
        return dataset

    total_processed = 0
    for files, is_parquet in ((parquet_files, True), (jsonl_files, False)):
        dataset = iter_dataset(files, is_parquet)
        if dataset is None:
            continue

        total = len(dataset)
        if args.limit is not None:
            total = min(total, args.limit)

        for idx in range(total):
            sample = dataset[idx]
            messages = sample.get("raw_prompt", None)
            if messages is None:
                # fallback: reconstruct minimal messages from question
                question = sample.get("question", "")
                messages = [
                    {"role": "system", "content": "Answer the question using <search> if needed and finalize with <answer>."},
                    {"role": "user", "content": question},
                ]

            try:
                result = await agent.run_sequential(messages, sampling_params)
                final_results.append(result)

                # optionally write decoded output
                if output_fp is not None:
                    try:
                        text = tokenizer.decode(result.response_ids, skip_special_tokens=False)
                    except Exception:
                        text = ""
                    out_line = {
                        "input": sample.get("question", ""),
                        "output": text,
                        "groundtruth": sample.get("golden_answers", []),
                    }
                    output_fp.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                    output_fp.flush()
            except Exception as e:
                # Log per-sample error and continue
                if output_fp is not None:
                    out_line = {
                        "input": sample.get("question", ""),
                        "output": "",
                        "groundtruth": sample.get("golden_answers", []),
                        "error": f"{e.__class__.__name__}: {str(e)}",
                    }
                    output_fp.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                    output_fp.flush()
                else:
                    print(f"Sample {idx} failed: {e}")

            total_processed += 1
            if total_processed % 10 == 0:
                print(f"Processed {total_processed} samples...")

    if output_fp is not None:
        output_fp.close()


if __name__ == "__main__":

    asyncio.run(_async_main())
    import pdb
    pdb.set_trace()

    print(1)
    
