"""
ScienceWorld Environment Server

Serves multiple ScienceWorld environments in parallel with resource pool management.
When the pool is full, create requests block until a slot becomes available.
Ray workers (or any client) can request envs by seed, interact via env_id, and release when done.

Usage:
    python -m tasks.classic_games.scienceworld.scienceworld_env_server --host 0.0.0.0 --port 8765 --pool-size 2
"""

import asyncio
import logging
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

try:
    import psutil
except ImportError:
    psutil = None
from pydantic import BaseModel

# Add project root to path
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tasks.classic_games.scienceworld.scienceworld_env import ScienceWorldEnv_Wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ScienceWorldEnvServer")

T = TypeVar("T")


class PriorityScheduler:
    """
    Schedules requests by priority. High-priority (ops on existing envs) run before
    low-priority (create). Multiple workers process the queues concurrently.
    """

    def __init__(self, num_workers: int = 16):
        self._high_queue: asyncio.Queue = asyncio.Queue()
        self._low_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []
        self._num_workers = num_workers

    async def start(self):
        if self._running:
            return
        self._running = True
        self._worker_tasks = [
            asyncio.create_task(self._worker()) for _ in range(self._num_workers)
        ]
        logger.info(
            "Priority scheduler started (ops before create, %d workers)",
            self._num_workers,
        )

    async def stop(self):
        self._running = False
        for t in self._worker_tasks:
            t.cancel()
        for t in self._worker_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._worker_tasks = []

    async def submit_high(self, coro: Coroutine[Any, Any, T]) -> T:
        """Submit high-priority job (step, reset, reset_state, observe, get_key_stats, close)."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._high_queue.put((coro, future))
        return await future

    async def submit_low(self, coro: Coroutine[Any, Any, T]) -> T:
        """Submit low-priority job (create)."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._low_queue.put((coro, future))
        return await future

    def _get_next_job(self) -> Optional[tuple]:
        """Get next job: high-priority first, else low-priority (non-blocking)."""
        try:
            return self._high_queue.get_nowait()
        except asyncio.QueueEmpty:
            try:
                return self._low_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

    async def _worker(self):
        while self._running:
            coro, future = None, None
            try:
                job = self._get_next_job()
                if job is None:
                    await asyncio.sleep(0.001)
                    continue
                coro, future = job
                if coro is not None and future is not None:
                    try:
                        result = await coro
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Scheduler worker error: %s", e)
                if future is not None and not future.done():
                    future.set_exception(e)


scheduler: Optional[PriorityScheduler] = None


class EnvConfig:
    max_steps = 30


# --- Request/Response models ---
class EnvConfigPayload(BaseModel):
    """Optional env config. Omitted fields use server defaults (max_steps=30)."""
    max_steps: Optional[int] = None


class CreateEnvRequest(BaseModel):
    seed: Optional[int] = None
    specified_game_idx: Optional[int] = None
    env_config: Optional[EnvConfigPayload] = None


class CreateEnvResponse(BaseModel):
    env_id: str


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    specified_game_idx: Optional[int] = None


class StepRequest(BaseModel):
    action: str


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict


class ResetStateRequest(BaseModel):
    game_idx: int
    full_action_history: list


# --- Server state ---
envs: dict[str, ScienceWorldEnv_Wrapper] = {}
pool_semaphore: Optional[asyncio.Semaphore] = None
executor: Optional[ThreadPoolExecutor] = None
pool_size: int = 4


def _create_env_blocking(
    env_config_dict: Optional[dict] = None,
) -> ScienceWorldEnv_Wrapper:
    """Blocking env creation - runs in thread pool."""
    defaults = {"max_steps": getattr(EnvConfig, "max_steps", 30)}
    if env_config_dict:
        defaults.update({k: v for k, v in env_config_dict.items() if v is not None})
    config = type("EnvConfig", (), defaults)()
    env = ScienceWorldEnv_Wrapper(config)
    # env.reset(specified_game_idx=specified_game_idx)
    return env


def _reset_env_blocking(env: ScienceWorldEnv_Wrapper, specified_game_idx: Optional[int] = None):
    env.reset(specified_game_idx=specified_game_idx)


def _step_env_blocking(env: ScienceWorldEnv_Wrapper, action: str):
    return env.step(action)


def _reset_state_blocking(env: ScienceWorldEnv_Wrapper, state_info: dict):
    return env.reset_state(state_info)


def _close_env_blocking(env: ScienceWorldEnv_Wrapper):
    """Close env, suppressing noisy py4j shutdown errors (Gateway not connected, etc.)."""
    import logging
    py4j_logger = logging.getLogger("py4j.java_gateway")
    old_level = py4j_logger.level
    py4j_logger.setLevel(logging.WARNING)
    try:
        try:
            env.close()
        except Exception as e:
            logger.warning(f"Error closing env: {e}")
    finally:
        py4j_logger.setLevel(old_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool_semaphore, executor, scheduler
    pool_semaphore = asyncio.Semaphore(pool_size)
    executor = ThreadPoolExecutor(max_workers=pool_size * 2)
    scheduler = PriorityScheduler(num_workers=pool_size * 2)
    await scheduler.start()
    logger.info(f"Server started with pool_size={pool_size}")
    yield
    await scheduler.stop()
    for eid, env in list(envs.items()):
        try:
            executor.submit(_close_env_blocking, env)
        except Exception as e:
            logger.warning(f"Cleanup env {eid}: {e}")
    envs.clear()
    executor.shutdown(wait=False)
    logger.info("Server shutdown complete")


app = FastAPI(title="ScienceWorld Env Server", lifespan=lifespan)


async def _do_create_env(req: CreateEnvRequest) -> CreateEnvResponse:
    """Low-priority: create env (needs new slot)."""
    await pool_semaphore.acquire()
    try:
        env_config_dict = None
        if req.env_config is not None:
            env_config_dict = req.env_config.model_dump(exclude_none=True)
        loop = asyncio.get_event_loop()
        env = await loop.run_in_executor(
            executor,
            _create_env_blocking,
            env_config_dict,
        )
        env_id = str(uuid.uuid4())
        envs[env_id] = env
        logger.info(f"Created env {env_id} (pool: {len(envs)}/{pool_size})")
        return CreateEnvResponse(env_id=env_id)
    except Exception:
        pool_semaphore.release()
        raise


@app.post("/create", response_model=CreateEnvResponse)
async def create_env(req: CreateEnvRequest):
    """Create a new env. Low-priority: runs after step/close so slots free faster."""
    try:
        coro = _do_create_env(req)
        return await scheduler.submit_low(coro)
    except Exception as e:
        logger.exception("Failed to create env: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _do_reset_env(env_id: str, req: ResetRequest):
    if env_id not in envs:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")
    env = envs[env_id]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        executor,
        _reset_env_blocking,
        env,
        req.specified_game_idx,
    )
    obs = env.observe()
    task_description = getattr(env, "task_goal", "") or ""
    info_serializable = {}
    if hasattr(env, "last_info") and env.last_info:
        for k, v in env.last_info.items():
            try:
                import json
                json.dumps(v)
                info_serializable[k] = v
            except (TypeError, ValueError):
                info_serializable[k] = str(v)
    return {"observation": obs, "task_description": task_description, "info": info_serializable}


@app.post("/reset/{env_id}")
async def reset_env(env_id: str, req: ResetRequest):
    """High-priority: runs before create so envs make progress."""
    return await scheduler.submit_high(_do_reset_env(env_id, req))


async def _do_reset_state_env(env_id: str, req: ResetStateRequest):
    if env_id not in envs:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")
    env = envs[env_id]
    state_info = {"game_idx": req.game_idx, "full_action_history": req.full_action_history}
    loop = asyncio.get_event_loop()
    obs = await loop.run_in_executor(executor, _reset_state_blocking, env, state_info)
    task_description = getattr(env, "task_goal", "") or ""
    info_serializable = {}
    if hasattr(env, "last_info") and env.last_info:
        for k, v in env.last_info.items():
            try:
                import json
                json.dumps(v)
                info_serializable[k] = v
            except (TypeError, ValueError):
                info_serializable[k] = str(v)
    return {"observation": obs, "task_description": task_description, "info": info_serializable}


@app.post("/reset_state/{env_id}")
async def reset_state_env(env_id: str, req: ResetStateRequest):
    """High-priority: runs before create."""
    return await scheduler.submit_high(_do_reset_state_env(env_id, req))


async def _do_step_env(env_id: str, req: StepRequest) -> StepResponse:
    if env_id not in envs:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")
    env = envs[env_id]
    loop = asyncio.get_event_loop()
    obs, reward, done, info = await loop.run_in_executor(
        executor,
        _step_env_blocking,
        env,
        req.action,
    )
    # Serialize info - some values may not be JSON-serializable
    info_serializable = {}
    if info:
        for k, v in info.items():
            try:
                import json
                json.dumps(v)
                info_serializable[k] = v
            except (TypeError, ValueError):
                info_serializable[k] = str(v)
    return StepResponse(observation=obs, reward=reward, done=done, info=info_serializable)


@app.post("/step/{env_id}", response_model=StepResponse)
async def step_env(env_id: str, req: StepRequest):
    """High-priority: runs before create so envs make progress and close faster."""
    return await scheduler.submit_high(_do_step_env(env_id, req))


async def _do_observe_env(env_id: str):
    if env_id not in envs:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")
    return {"observation": envs[env_id].observe()}


@app.get("/observe/{env_id}")
async def observe_env(env_id: str):
    """High-priority: runs before create."""
    return await scheduler.submit_high(_do_observe_env(env_id))


async def _do_get_key_stats(env_id: str):
    if env_id not in envs:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")
    return envs[env_id].get_key_stats()


@app.get("/get_key_stats/{env_id}")
async def get_key_stats(env_id: str):
    """High-priority: runs before create."""
    return await scheduler.submit_high(_do_get_key_stats(env_id))


async def _do_close_env(env_id: str):
    if env_id not in envs:
        raise HTTPException(status_code=404, detail=f"Env {env_id} not found")
    env = envs.pop(env_id)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, _close_env_blocking, env)
    except Exception as e:
        logger.warning("Error during close of env %s: %s", env_id, e)
    finally:
        pool_semaphore.release()
    logger.info(f"Closed env {env_id} (pool: {len(envs)}/{pool_size})")
    return {"success": True}


@app.post("/close/{env_id}")
async def close_env(env_id: str):
    """High-priority: runs before create to free slots for blocked create requests."""
    return await scheduler.submit_high(_do_close_env(env_id))


@app.get("/pool_status")
async def pool_status():
    high_pending = scheduler._high_queue.qsize() if scheduler else 0
    low_pending = scheduler._low_queue.qsize() if scheduler else 0
    return {
        "active_envs": len(envs),
        "pool_size": pool_size,
        "available_slots": pool_size - len(envs),
        "pending_high": high_pending,
        "pending_create": low_pending,
    }


def _build_env_progress_display(env_list: list[dict], bar_width: int = 30) -> str:
    """Build ASCII progress bar display for terminal output."""
    if not env_list:
        return "No active environments.\n"
    lines = [
        "Env Progress (ranked by steps, high → low)",
        "=" * 60,
    ]
    for i, e in enumerate(env_list, 1):
        step = e["current_step"]
        max_s = e["max_steps"]
        pct = step / max_s if max_s > 0 else 0
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        env_short = (e["env_id"][:8] + "…") if len(e["env_id"]) > 8 else e["env_id"]
        game_idx = e.get("game_idx", "?")
        status = e.get("game_status") or "running"
        lines.append(f"#{i} {env_short} [{bar}] {step:2d}/{max_s} (game_idx:{game_idx}, {status})")
    lines.append("")
    return "\n".join(lines)


@app.get("/env_status")
async def env_status(format: Optional[str] = None):
    """
    Detailed status of each active environment: step count, progress bar, ranked high→low.
    Use `curl .../env_status` for JSON, or `curl .../env_status?format=text` for ASCII bars.
    For JSON + display: `curl -s .../env_status | jq -r '.display'`
    """
    if not envs:
        if format == "text":
            return Response(content="No active environments.\n", media_type="text/plain")
        return {
            "active_envs": 0,
            "envs": [],
            "display": "No active environments.\n",
        }

    env_list = []
    for env_id, env in envs.items():
        step = getattr(env, "current_step", 0)
        max_s = getattr(env, "max_steps", 30)
        game_idx = getattr(env, "specific_game_idx", None)
        game_status = getattr(env, "game_status", None)
        env_list.append({
            "env_id": env_id,
            "current_step": step,
            "max_steps": max_s,
            "game_idx": game_idx,
            "game_status": game_status,
        })

    env_list.sort(key=lambda x: x["current_step"], reverse=True)

    display = _build_env_progress_display(env_list)

    if format == "text":
        return Response(content=display, media_type="text/plain")

    return {
        "active_envs": len(env_list),
        "envs": env_list,
        "display": display,
    }


def _get_resource_usage() -> dict[str, Any]:
    """Get server and Java-related resource usage."""
    result: dict[str, Any] = {
        "server": None,
        "java": {"count": 0, "total_rss_mb": 0.0, "total_cpu_percent": 0.0, "processes": []},
        "psutil_available": psutil is not None,
    }
    if psutil is None:
        return result

    try:
        # Server (Python) process
        proc = psutil.Process()
        mem = proc.memory_info()
        result["server"] = {
            "pid": proc.pid,
            "rss_mb": mem.rss / (1024 * 1024),
            "cpu_percent": proc.cpu_percent(interval=0.1),
        }

        # Java processes: prefer server's descendants (ScienceWorld spawns Java as children)
        java_procs: list[dict] = []
        total_rss = 0.0
        total_cpu = 0.0

        def collect_java(p: "psutil.Process") -> None:
            nonlocal total_rss, total_cpu
            try:
                name = p.name().lower() if p.is_running() else ""
                if "java" in name:
                    mem_info = p.memory_info()
                    rss_mb = mem_info.rss / (1024 * 1024)
                    cpu = p.cpu_percent(interval=None)  # Non-blocking snapshot
                    java_procs.append({"pid": p.pid, "rss_mb": round(rss_mb, 2), "cpu_percent": round(cpu, 2)})
                    total_rss += rss_mb
                    total_cpu += cpu
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                pass

        # First: server's child processes (ScienceWorld Java processes)
        for child in proc.children(recursive=True):
            collect_java(child)

        # Fallback: all Java processes if no children (e.g. different process tree)
        if not java_procs:
            for p in psutil.process_iter(["pid", "name", "memory_info"]):
                try:
                    if "java" in (p.info.get("name") or "").lower():
                        mem_info = p.info.get("memory_info")
                        if mem_info:
                            rss_mb = mem_info.rss / (1024 * 1024)
                            total_rss += rss_mb
                            cpu = p.cpu_percent(interval=None) if hasattr(p, "cpu_percent") else 0
                            total_cpu += cpu
                            java_procs.append({"pid": p.pid, "rss_mb": round(rss_mb, 2), "cpu_percent": round(cpu, 2)})
                except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                    pass

        result["java"] = {
            "count": len(java_procs),
            "total_rss_mb": round(total_rss, 2),
            "total_cpu_percent": round(total_cpu, 2),
            "processes": java_procs[:20],  # Limit list size
        }
    except Exception as e:
        logger.warning(f"Error getting resource usage: {e}")
        result["error"] = str(e)

    return result


@app.get("/resource_usage")
async def resource_usage():
    """Report server and Java-related resource usage (memory, CPU)."""
    return _get_resource_usage()


def main():
    import argparse
    global pool_size
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pool-size", type=int, default=2, help="Max concurrent envs (resource limit)")
    args = parser.parse_args()
    pool_size = args.pool_size
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


    # python -m tasks.classic_games.scienceworld.scienceworld_env_server --host 127.0.0.1 --port 8765 --pool-size 2
