import asyncio
import base64
import os
import pdb
import re
import sys
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import ray
from datasets import load_dataset
from omegaconf import OmegaConf
from openai.types.chat.chat_completion import ChatCompletion
import random
from typing import Any, Dict, List, Union, Callable
from tensordict import TensorDict
import torch

from src.workers.rollout.vllm_rollout.scheduler.naive_chat_scheduler import NaiveChatCompletionScheduler
from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from transformers.utils.chat_template_utils import render_jinja_template
import verl.utils.torch_functional as verl_F
from verl.utils import hf_processor
from verl.utils.dataset.vision_utils import process_image
from verl.utils.model import compute_position_id_with_mask

from dataclasses import dataclass
from typing import Optional
import importlib

DEBUG=False


class MultiModalChatCompletionScheduler(NaiveChatCompletionScheduler):
    def __init__(self, config, model_path, server_addresses, **kwargs):
        """
        :param config: full scale config
        :param model_path:
        :param server_addresses:
        :param kwargs:
        """
        super().__init__(config, model_path, server_addresses, **kwargs)

        multi_turn_cfg = self.config.rollout.multi_turn
        self.each_response_max_length = multi_turn_cfg.each_response_length
        self.total_response_max_length = multi_turn_cfg.tool_response_length

        self.processor = hf_processor(model_path)


    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=1,  # self.config.n,   keep at 1, and create n envs
            max_completion_tokens=self.each_response_max_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=["</answer>", "</think>"],
            include_stop_str_in_output=True,
            # extra_body={
            #     "include_stop_str_in_output": True,
            #     "stop": ["</answer>", "</think>"],
            # },
        )

        demo_image_path = os.environ.get("MULTIMODAL_DEMO_IMAGE")
        if not demo_image_path:
            raise ValueError("Set MULTIMODAL_DEMO_IMAGE to a local image path before running multimodal demo generation.")
        if not os.path.exists(demo_image_path):
            raise FileNotFoundError(f"MULTIMODAL_DEMO_IMAGE does not exist: {demo_image_path}")

        img_1_url = {"image": demo_image_path}
        img_1_description = "A woman sits on the beach at sunset, smiling as she shares a high five with her large dog."
        # GitHub Logo
        img_2_url = {"image": "https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"}
        img_2_description = "A GitHub Logo image"
        # Octocat
        img_3_url = {"image": "https://octodex.github.com/images/orderedlistocat.png"}
        img_3_description = "An Octocat image"

        image_list = [img_1_url] #, img_2_url, img_3_url]
        description_list = [img_1_description, img_2_description, img_3_description]

        processed_images = []
        for img_url in image_list:
            img = process_image(img_url)
            processed_images.append(img)

        with open(img_1_url['image'], "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # placeholders = [{"type": "image", "image": url} for url in image_list]
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        }, {
            "role":
                "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe these three image in detail:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
                {"type": "text", "text": "Are you sure?"},
            ],
        }]



        prompt = self.processor.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        #
        print(f"prompt = {prompt}")
        #
        # inputs = {
        #     "prompt": prompt,
        #     "multi_modal_data": {
        #         "image": processed_images
        #     },
        # }

        async def dummy_fn(completions, callback_additional_info, exception):
            tmp_holder = callback_additional_info['tmp_holder']
            tmp_holder.append(completions.choices[0].message.content)
            # tmp_holder.append(completions)

            return

        tasks = []
        final_answer = []

        tasks.append(
            asyncio.create_task(
                self.submit_chat_completions(
                    callback=dummy_fn,
                    callback_additional_info={
                        "tmp_holder": final_answer,
                    },
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )
            )
        )

        await asyncio.gather(*tasks)

        print(f"final answer = {final_answer}")

        return
