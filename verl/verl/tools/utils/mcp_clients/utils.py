# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time

from mcp import Tool

logger = logging.getLogger(__file__)


class TokenBucket:
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit  # tokens per second
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            # Add new tokens based on time elapsed
            new_tokens = (now - self.last_update) * self.rate_limit
            self.tokens = min(self.rate_limit, self.tokens + new_tokens)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class RPMBucket:
    def __init__(self, rate_limit: float):
        self.RPM = rate_limit * 60      # request per minute
        self.rate_limit = rate_limit
        self.current_request = self.RPM
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()

            new_request = (now - self.last_update) * self.rate_limit
            self.current_request = min(self.RPM, self.current_request + new_request)
            self.last_update = now

            if self.current_request >= 1:
                self.current_request -= 1
                return True

            print(f"### Waiting for RPMBucket to be topped up... Current request budget {self.current_request}")
            return False







class CallBucket:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.max_calls = calls_per_minute
        self.calls = calls_per_minute  # Start with full bucket
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            # Calculate minutes elapsed since last update
            minutes_elapsed = (now - self.last_update) / 60.0
            
            # Add new calls based on time elapsed (calls per minute)
            new_calls = minutes_elapsed * self.calls_per_minute
            self.calls = min(self.max_calls, self.calls + new_calls)
            self.last_update = now

            if self.calls >= 1:
                self.calls -= 1
                return True
            return False

    def get_remaining_calls(self) -> int:
        """Get the number of remaining calls available"""
        with self.lock:
            now = time.time()
            minutes_elapsed = (now - self.last_update) / 60.0
            new_calls = minutes_elapsed * self.calls_per_minute
            current_calls = min(self.max_calls, self.calls + new_calls)
            return int(current_calls)

    def get_time_to_next_call(self) -> float:
        """Get time in seconds until next call is available"""
        with self.lock:
            if self.calls >= 1:
                return 0.0
            
            now = time.time()
            minutes_elapsed = (now - self.last_update) / 60.0
            new_calls = minutes_elapsed * self.calls_per_minute
            current_calls = min(self.max_calls, self.calls + new_calls)
            
            if current_calls >= 1:
                return 0.0
            
            # Calculate how many minutes we need to wait for 1 call
            calls_needed = 1 - current_calls
            minutes_needed = calls_needed / self.calls_per_minute
            return minutes_needed * 60.0


def mcp2openai(mcp_tool: Tool) -> dict:
    """Convert a MCP Tool to an OpenAI ChatCompletionTool."""
    openai_format = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
            "strict": False,
        },
    }
    if not openai_format["function"]["parameters"].get("required", None):
        openai_format["function"]["parameters"]["required"] = []
    return openai_format
