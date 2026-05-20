import asyncio
import os
from typing import Any, Dict, List

import torch
from openai.types.chat.chat_completion import ChatCompletion
from tensordict import TensorDict

from verl.protocol import DataProto




class BaseSyncScheduler:
    def __init__(
        self,
        config: DictConfig,
        model_path: str,

    ):
        """
        Args:
            config: DictConfig, rollout config.
            model_path: str, model path.
            server_addresses: List[str], server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.api_key = config.get("api_key", None) or os.environ.get("VLLM_ROLLOUT_API_KEY")

    def _get_api_key(self) -> str:
        if not self.api_key:
            raise ValueError("VLLM rollout API key is required. Set rollout.api_key or VLLM_ROLLOUT_API_KEY.")
        return self.api_key

    async def submit_chat_completions(
        self,
        callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None],
        callback_additional_info: Dict[str, Any],
        **chat_complete_request,
    ):
        """
        Submit a chat completion request to the server with the least number of requests.

        Args:
            callback: Callable[[ChatCompletion, Dict[str, Any], Exception], None], async callback function
                to handle the response. The callback function should have the following signature:

                ```python
                async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
                    ...
                ```
                - completions: chat completion response from server.
                - info: user provided `callback_additional_info`.
                - exception: exception raise from OpenAI client if request failed, otherwise None.

                **CAUTION**: the callback function must be async and non-blocking, if you have any blocking operation,
                please move to seperate thread or process pool to avoid blocking the event loop.

            callback_additional_info: Dict[str, Any], additional info to pass to the callback function.

            **chat_complete_request: dict, request parameters same as OpenAI AsyncCompletions.create.
                OpenAI API reference: https://platform.openai.com/docs/api-reference/chat/create
        """
        if "extra_headers" not in chat_complete_request:
            chat_complete_request["extra_headers"] = {}

        extra_headers = chat_complete_request["extra_headers"]
        request_id = extra_headers.get("x-request-id", None)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
                extra_headers["x-request-id"] = request_id

            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address
        chat_complete_request["extra_headers"]["x-request-id"] = request_id

        completions, exception = None, None
        try:
            # NOTE: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            completions = await self._chat_completions_aiohttp(address, **chat_complete_request)
        except Exception as e:
            # Let user handle the exception
            exception = e

        await callback(completions, callback_additional_info, exception)

    async def _chat_completions_openai(self, address: str, **chat_complete_request) -> ChatCompletion:
        client = AsyncOpenAI(base_url=f"http://{address}/v1", api_key=self._get_api_key(), timeout=None, max_retries=0)
        return await client.chat.completions.create(**chat_complete_request)

    async def _chat_completions_aiohttp(self, address: str, **chat_complete_request) -> ChatCompletion:
        try:
            extra_headers = chat_complete_request.pop("extra_headers")
            timeout = aiohttp.ClientTimeout(total=None)
            session = aiohttp.ClientSession(timeout=timeout)
            async with session.post(
                url=f"http://{address}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._get_api_key()}", **extra_headers},
                json=chat_complete_request,
            ) as resp:
                data = await resp.json()
                return ChatCompletion(**data)
        finally:
            await session.close()

    async def generate_sequences(self, prompts: DataProto, **sampling_params) -> DataProto:
        raise NotImplementedError
