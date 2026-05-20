"""
ScienceWorld Environment Client

Client for Ray workers (or any process) to interact with the ScienceWorld env server.
Each client holds an env_id after create(), and all subsequent calls use that env_id.

Provides both sync (ScienceWorldEnvClient) and async (AsyncScienceWorldEnvClient) clients.
"""

import asyncio
import time
from typing import Any, Optional

try:
    import httpx
except ImportError:
    httpx = None

try:
    import requests
except ImportError:
    requests = None


def _env_config_to_dict(obj: Any) -> Any:
    """Convert env_config to plain Python types for JSON serialization (handles OmegaConf DictConfig, ListConfig)."""
    if obj is None:
        return None
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    # Fallback without omegaconf
    if isinstance(obj, dict):
        return {k: _env_config_to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "items") and callable(getattr(obj, "items")) and not isinstance(obj, (list, tuple)):
        return {str(k): _env_config_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_env_config_to_dict(x) for x in obj]
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict)):
        try:
            return [_env_config_to_dict(x) for x in obj]
        except TypeError:
            return obj
    return obj


class ScienceWorldEnvClient:
    """Client for interacting with ScienceWorld env server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        timeout: float = 60.0,
        env_config: Optional[dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.env_config = env_config  # e.g. {"max_steps": 50}; sent to server on create()
        self.env_id: Optional[str] = None
        self._session = None

        self.num_train = 2294
        self.num_test = 1308
        self.env_created = False

        # create
        _ = self.create()

    @property
    def num_game(self):
        return self.num_train + self.num_test

    @property
    def legal_moves_list(self):
        return []

    @property
    def legal_moves(self):
        return self.legal_moves_list
    
    @property
    def legal_moves_string(self):
        return "[" + ";\n ".join(self.legal_moves_list) + "]"

    def _get_session(self):
        if httpx is not None:
            if self._session is None:
                self._session = httpx.Client(base_url=self.base_url, timeout=self.timeout)
            return self._session
        elif requests is not None:
            return requests
        raise ImportError("Need httpx or requests. Install: pip install httpx")

    def _request(self, method: str, path: str, max_retries: int = 3, **kwargs) -> dict:
        """Make HTTP request with retry on transient connection errors (e.g. connection reset by peer)."""
        last_exc = None
        for attempt in range(max_retries):
            try:
                if httpx is not None:
                    if self._session is None:
                        self._session = httpx.Client(base_url=self.base_url, timeout=self.timeout)
                    r = self._session.request(method, path, **kwargs)
                elif requests is not None:
                    url = f"{self.base_url}{path}"
                    r = requests.request(method, url, timeout=self.timeout, **kwargs)
                else:
                    raise ImportError("Need httpx or requests. Install: pip install httpx")
                r.raise_for_status()
                return r.json() if r.content else {}
            except Exception as e:
                retryable = isinstance(e, (ConnectionError, OSError))
                if httpx is not None:
                    retryable = retryable or isinstance(e, (httpx.ReadError, httpx.ConnectError))
                if not retryable:
                    raise
                last_exc = e
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    raise
        raise last_exc

    def create(
        self,
    ) -> str:
        """Create a new env on the server. Blocks until a pool slot is available."""
        payload = {}

        if self.env_config is not None:
            payload["env_config"] = _env_config_to_dict(self.env_config)
        resp = self._request("POST", "/create", json=payload)
        self.env_id = resp["env_id"]
        self.env_created = True

        return self.env_id

    def reset(
        self,
        seed: Optional[int] = None,
        specified_game_idx: Optional[int] = None,
    ) -> str:
        """Reset the env. Returns observation."""
    
        if not self.env_created:
            self.create()

        if self.env_id is None:
            raise RuntimeError("No env created. Call create() first.")
        payload = {}
        if seed is not None:
            payload["seed"] = seed
        if specified_game_idx is not None:
            payload["specified_game_idx"] = specified_game_idx
        resp = self._request("POST", f"/reset/{self.env_id}", json=payload)
        self.game_status = None
        self.last_obs = resp["observation"]
        self.specific_game_idx = specified_game_idx
        self.last_info = resp['info']
        self.task_goal = resp.get("task_description", "")
        self.full_action_history = []
        self.current_step = 0
        self.reward_record = 0
        self.final_reward = 0

        return self.observe()

    def reset_state(self, state_info) -> str:
        """Reset env to a specific state. Returns observation."""
        
        if not self.env_created:
            self.create()

        if self.env_id is None:
            raise RuntimeError("No env created. Call create() first.")

        resp = self._request("POST", f"/reset_state/{self.env_id}", json=state_info)
        self.game_status = None
        self.last_obs = resp["observation"]
        self.task_goal = state_info["task_description"]
        self.specific_game_idx = state_info["game_idx"]
        self.last_info = resp['info']
        self.full_action_history = []
        self.current_step = 0
        self.reward_record = 0
        self.final_reward = 0
        
        return self.observe()

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Step the env. Returns (observation, reward, done, info)."""
        if self.env_id is None:
            raise RuntimeError("No env created. Call create() first.")
        resp = self._request("POST", f"/step/{self.env_id}", json={"action": action})
        self.game_status = resp['info']['game_status']
        self.last_obs = resp["observation"]
        self.last_info = resp['info']
        self.current_step += 1
        reward = resp['reward']
        if reward > self.reward_record:
            self.reward_record = reward

        return (
            resp["observation"],
            resp["reward"],
            resp["done"],
            resp.get("info", {}),
        )

    def observe(self) -> str:
        return self.last_obs

    def get_key_stats(self) -> dict:
        return {
            "game_idx": self.specific_game_idx,
            "last_obs": self.last_obs,
            "last_info": self.last_info,
            "task_description": self.task_goal,
            "full_action_history": [i for i in self.full_action_history],
        }
    
    def check_success(self):
        return self.game_status == "win"
    
    def check_lose(self):
        return self.game_status == "lose"
    
    def check_results(self):
        return self.game_status

    def close(self):
        """Release the env back to the pool."""
        if self.env_id is None:
            return
        # try:
        self._request("POST", f"/close/{self.env_id}")
        # finally:
        self.env_id = None
        if self._session is not None and httpx is not None:
            self._session.close()
            self._session = None

    def pool_status(self) -> dict:
        """Get server pool status (for debugging)."""
        return self._request("GET", "/pool_status")

    def resource_usage(self) -> dict:
        """Get server and Java-related resource usage (memory, CPU)."""
        return self._request("GET", "/resource_usage")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncScienceWorldEnvClient:
    """Async client for interacting with ScienceWorld env server. Requires httpx."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        timeout: float = 300.0,
        env_config: Optional[dict[str, Any]] = None,
    ):
        if httpx is None:
            raise ImportError("AsyncScienceWorldEnvClient requires httpx. Install: pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.env_config = env_config  # e.g. {"max_steps": 50}; sent to server on create()
        self.env_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

        self.num_train = 2294
        self.num_test = 1308
        self.env_created = False

    @classmethod
    async def connect(
        cls,
        base_url: str = "http://127.0.0.1:8765",
        timeout: float = 300.0,
        env_config: Optional[dict[str, Any]] = None,
    ) -> "AsyncScienceWorldEnvClient":
        """Create client and connect to server (create env). Use instead of __init__ when async init is needed."""
        client = cls(base_url=base_url, timeout=timeout, env_config=env_config)
        await client.create()
        return client

    @property
    def num_game(self):
        return self.num_train + self.num_test

    @property
    def legal_moves_list(self):
        return []

    @property
    def legal_moves(self):
        return self.legal_moves_list
    
    @property
    def legal_moves_string(self):
        return "[" + ";\n ".join(self.legal_moves_list) + "]"

    # @property
    # def legal_moves_string(self):
    #     return ", ".join(self.legal_moves_list)
  
    async def _request(self, method: str, path: str, max_retries: int = 3, **kwargs) -> dict:
        """Make HTTP request with retry on transient connection errors."""
        last_exc = None
        for attempt in range(max_retries):
            try:
                if self._client is None:
                    self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
                r = await self._client.request(method, path, **kwargs)
                r.raise_for_status()
                return r.json() if r.content else {}
            except Exception as e:
                retryable = isinstance(e, (ConnectionError, OSError))
                if httpx is not None:
                    retryable = retryable or isinstance(e, (httpx.ReadError, httpx.ConnectError))
                if not retryable:
                    raise
                last_exc = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    raise
        raise last_exc

    async def create(
        self,
    ) -> str:
        """Create a new env on the server. Blocks until a pool slot is available."""
        payload = {}
        if self.env_config is not None:
            payload["env_config"] = _env_config_to_dict(self.env_config)
        resp = await self._request("POST", "/create", json=payload)
        self.env_id = resp["env_id"]
        self.env_created = True
        return self.env_id

    async def reset(
        self,
        seed: Optional[int] = None,
        specified_game_idx: Optional[int] = None,
    ) -> str:
        """Reset the env. Returns observation."""
        if not self.env_created:
            await self.create()
        
        if self.env_id is None:
            raise RuntimeError("No env created. Call create() first.")
        payload = {}
        if seed is not None:
            payload["seed"] = seed
        if specified_game_idx is not None:
            payload["specified_game_idx"] = specified_game_idx
        resp = await self._request("POST", f"/reset/{self.env_id}", json=payload)
        self.game_status = None
        self.last_obs = resp["observation"]
        self.specific_game_idx = specified_game_idx
        self.last_info = resp['info']
        self.task_goal = resp.get("task_description", "")
        self.full_action_history = []
        self.current_step = 0
        self.reward_record = 0
        self.final_reward = 0

        return self.observe()

    async def reset_state(self, state_info: dict) -> str:
        """Reset env to a specific state. Returns observation."""
        if not self.env_created:
            await self.create()

        if self.env_id is None:
            raise RuntimeError("No env created. Call create() first.")
        payload = state_info

        resp = await self._request("POST", f"/reset_state/{self.env_id}", json=payload)

        self.game_status = None
        self.last_obs = resp["observation"]
        self.task_goal = state_info["task_description"]
        self.specific_game_idx = state_info["game_idx"]
        self.last_info = resp['info']
        self.full_action_history = []
        self.current_step = 0
        self.reward_record = 0
        self.final_reward = 0

        return self.observe()

    async def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Step the env. Returns (observation, reward, done, info)."""
        if self.env_id is None:
            raise RuntimeError("No env created. Call create() first.")
        resp = await self._request("POST", f"/step/{self.env_id}", json={"action": action})
        self.game_status = resp['info']['game_status']

        self.last_obs = resp["observation"]
        self.last_info = resp['info']
        self.current_step += 1
        reward = resp['reward']
        if reward > self.reward_record:
            self.reward_record = reward

        return (
            resp["observation"],
            resp["reward"],
            resp["done"],
            resp.get("info", {}),
        )

    def observe(self) -> str:
        return self.last_obs

    def get_key_stats(self) -> dict:
        """Get key stats (game_idx, last_obs, etc.)."""
        # if self.env_id is None:
        #     raise RuntimeError("No env created. Call create() first.")
        # return await self._request("GET", f"/get_key_stats/{self.env_id}")
        return {
            "game_idx": self.specific_game_idx,
            "last_obs": self.last_obs,
            "last_info": self.last_info,
            "task_description": self.task_goal,
            "full_action_history": [i for i in self.full_action_history],
        }

    def check_success(self):
        return self.game_status == "win"
    
    def check_lose(self):
        return self.game_status == "lose"
    
    def check_results(self):
        return self.game_status

    async def close(self):
        """Release the env back to the pool."""
        if self.env_id is None:
            return
        try:
            await self._request("POST", f"/close/{self.env_id}")
        finally:
            self.env_id = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def pool_status(self) -> dict:
        """Get server pool status (for debugging)."""
        return await self._request("GET", "/pool_status")

    async def resource_usage(self) -> dict:
        """Get server and Java-related resource usage (memory, CPU)."""
        return await self._request("GET", "/resource_usage")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


async def wait_for_server_async(
    base_url: str = "http://127.0.0.1:8765", timeout: float = 30.0, poll_interval: float = 0.5
) -> bool:
    """Wait until the server is reachable (async). Requires httpx."""
    if httpx is None:
        raise ImportError("wait_for_server_async requires httpx. Install: pip install httpx")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                r = await client.get(f"{base_url.rstrip('/')}/pool_status")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(poll_interval)
    return False


def wait_for_server(base_url: str = "http://127.0.0.1:8765", timeout: float = 30.0, poll_interval: float = 0.5) -> bool:
    """Wait until the server is reachable."""
    import urllib.request
    import urllib.error
    url = f"{base_url.rstrip('/')}/pool_status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as r:
                r.read()
            return True
        except Exception:
            time.sleep(poll_interval)
    return False
