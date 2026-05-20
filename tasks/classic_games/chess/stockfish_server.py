import argparse
import asyncio
import importlib
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    chess = importlib.import_module("chess")
    importlib.import_module("chess.engine")
except ImportError as exc:  # pragma: no cover - required dependency check
    raise RuntimeError("python-chess must be installed to use the Stockfish tools.") from exc

logger = logging.getLogger(__name__)

class EnginePathNotFoundError(FileNotFoundError):
    """Raised when the Stockfish executable cannot be located."""


def resolve_engine_path(preferred_path: Optional[str] = None) -> str:
    """Find a usable Stockfish binary from explicit input, env, or PATH."""

    candidates: List[str] = []
    if preferred_path:
        candidates.append(preferred_path)

    env_candidate = os.environ.get("STOCKFISH_EXECUTABLE")
    if env_candidate:
        candidates.append(env_candidate)

    path_candidate = shutil.which("stockfish")
    if path_candidate:
        candidates.append(path_candidate)

    for path in candidates:
        if path and os.path.exists(path):
            logger.debug("Resolved Stockfish executable: %s", path)
            return path

    raise EnginePathNotFoundError(
        "Could not locate a Stockfish executable."
        " Provide a path, set STOCKFISH_EXECUTABLE, or install stockfish on PATH."
    )


class StockFish_Tool:
    """Backwards-compatible single Stockfish engine helper."""

    def __init__(
        self,
        engine_path: Optional[str] = None,
        engine_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.engine_path = engine_path
        self.engine_options = engine_options or {}
        self.engine: Optional[chess.engine.SimpleEngine] = None

    def load_engine(self) -> chess.engine.SimpleEngine:
        path = resolve_engine_path(self.engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        if self.engine_options:
            self.engine.configure(self.engine_options)
        return self.engine

    def infer_action(self, board: chess.Board, n: int = 1):
        if self.engine is None:
            raise RuntimeError("Please initialize the backend engine first via load_engine().")
        if n == 1:
            stockfish_result = self.engine.play(board, chess.engine.Limit(time=0.1))
            return stockfish_result.move
        if n > 1:
            return self.engine.analyse(board, chess.engine.Limit(depth=18), multipv=n)
        raise ValueError("n must be a positive integer.")

    def infer_state_score(self, board: chess.Board) -> int:
        if self.engine is None:
            raise RuntimeError("Please initialize the backend engine first via load_engine().")

        stockfish_info = self.engine.analyse(board, chess.engine.Limit(depth=20))
        relative = stockfish_info["score"].relative
        if isinstance(relative, chess.engine.Mate):
            return 2000
        return relative.score()

    def infer_analysis(self, board: chess.Board) -> Dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("Please initialize the backend engine first via load_engine().")
        return self.engine.analyse(board, chess.engine.Limit(depth=20))

    def quit(self) -> None:
        if self.engine:
            self.engine.quit()
            self.engine = None


@dataclass
class _EngineWrapper:
    identifier: int
    engine: chess.engine.SimpleEngine
    total_requests: int = 0
    last_used_at: float = field(default_factory=time.time)


class StockfishEnginePool:
    """Manage a pool of Stockfish engine instances for concurrent requests."""

    def __init__(
        self,
        pool_size: int = 4,
        engine_path: Optional[str] = None,
        engine_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if pool_size < 1:
            raise ValueError("Pool size must be at least 1.")
        self.pool_size = pool_size
        self.engine_path = engine_path
        self.engine_options = engine_options or {}

        self._available: Optional[asyncio.Queue[_EngineWrapper]] = None
        self._wrappers: List[_EngineWrapper] = []
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return

            path = resolve_engine_path(self.engine_path)
            self._available = asyncio.Queue(maxsize=self.pool_size)
            self._wrappers = []

            for idx in range(self.pool_size):
                wrapper = await asyncio.to_thread(self._create_wrapper, idx, path)
                self._wrappers.append(wrapper)
                await self._available.put(wrapper)

            self._started = True
            logger.info("Initialized Stockfish pool with %d engines using %s", self.pool_size, path)

    async def stop(self) -> None:
        async with self._lock:
            if not self._started:
                return

            for wrapper in self._wrappers:
                try:
                    wrapper.engine.quit()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to shut down engine %d cleanly: %s", wrapper.identifier, exc)

            if self._available:
                while not self._available.empty():
                    try:
                        self._available.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            self._wrappers.clear()
            self._available = None
            self._started = False
            logger.info("Stockfish engine pool stopped.")

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def available(self) -> int:
        if not self._started or self._available is None:
            return 0
        return self._available.qsize()

    def status(self) -> Dict[str, Any]:
        engines_status = []
        for wrapper in self._wrappers:
            engines_status.append(
                {
                    "engine_id": wrapper.identifier,
                    "total_requests": wrapper.total_requests,
                    "last_used": datetime.fromtimestamp(wrapper.last_used_at).isoformat(),
                }
            )
        return {
            "pool_size": self.pool_size,
            "started": self._started,
            "available": self.available,
            "engines": engines_status,
        }

    def _create_wrapper(self, identifier: int, path: str) -> _EngineWrapper:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        if self.engine_options:
            engine.configure(self.engine_options)
        return _EngineWrapper(identifier=identifier, engine=engine)

    async def _acquire(self, timeout: Optional[float] = None) -> _EngineWrapper:
        if not self._started or self._available is None:
            raise RuntimeError("Engine pool has not been started. Call await start() first.")

        try:
            if timeout is None:
                wrapper = await self._available.get()
            else:
                wrapper = await asyncio.wait_for(self._available.get(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Timed out waiting for a free Stockfish engine.") from exc

        return wrapper

    async def _release(self, wrapper: _EngineWrapper) -> None:
        if not self._started or self._available is None:
            return
        wrapper.last_used_at = time.time()
        await self._available.put(wrapper)

    @asynccontextmanager
    async def lease(self, timeout: Optional[float] = None) -> AsyncGenerator[_EngineWrapper, None]:
        wrapper = await self._acquire(timeout=timeout)
        try:
            yield wrapper
        finally:
            await self._release(wrapper)

    async def play(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        ponder: bool = False,
        timeout: Optional[float] = None,
    ) -> Tuple[chess.engine.PlayResult, int]:
        async with self.lease(timeout=timeout) as wrapper:
            board_copy = board.copy(stack=False)
            result = await asyncio.to_thread(wrapper.engine.play, board_copy, limit, ponder=ponder)
            wrapper.total_requests += 1
            return result, wrapper.identifier

    async def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: int = 1,
        info: Any = chess.engine.INFO_ALL,
        timeout: Optional[float] = None,
    ) -> Tuple[Any, int]:
        async with self.lease(timeout=timeout) as wrapper:
            board_copy = board.copy(stack=False)
            result = await asyncio.to_thread(
                wrapper.engine.analyse,
                board_copy,
                limit,
                info=info,
                multipv=multipv,
            )
            wrapper.total_requests += 1
            return result, wrapper.identifier


try:
    fastapi_module = importlib.import_module("fastapi")
    FastAPI = fastapi_module.FastAPI  # type: ignore[attr-defined]
    HTTPException = fastapi_module.HTTPException  # type: ignore[attr-defined]
    pydantic_module = importlib.import_module("pydantic")
    BaseModel = pydantic_module.BaseModel  # type: ignore[attr-defined]
    Field = pydantic_module.Field  # type: ignore[attr-defined]
    validator = pydantic_module.validator  # type: ignore[attr-defined]
except ImportError:
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    validator = None  # type: ignore[assignment]


if BaseModel is not None:

    class BaseStockfishRequest(BaseModel):
        fen: str = Field(..., description="Forsyth–Edwards Notation representing the board state.")
        time_limit: Optional[float] = Field(default=None, ge=0.0, description="Seconds to think.")
        depth: Optional[int] = Field(default=None, ge=1, description="Search depth limit.")
        nodes: Optional[int] = Field(default=None, ge=1, description="Node count limit.")

        @validator("fen")
        def validate_fen(cls, value: str) -> str:  # noqa: N805 (pydantic validator name style)
            try:
                chess.Board(value)
            except ValueError as exc:
                raise ValueError(f"Invalid FEN string: {exc}") from exc
            return value

        def to_limit(self) -> chess.engine.Limit:
            kwargs: Dict[str, Any] = {}
            if self.time_limit is not None:
                kwargs["time"] = self.time_limit
            if self.depth is not None:
                kwargs["depth"] = self.depth
            if self.nodes is not None:
                kwargs["nodes"] = self.nodes
            if not kwargs:
                kwargs["time"] = 0.1
            return chess.engine.Limit(**kwargs)


    class PlayRequest(BaseStockfishRequest):
        ponder: bool = Field(default=False, description="Request ponder move from the engine.")
        include_san: bool = Field(default=False, description="Include SAN notation for the best move.")


    class AnalyseRequest(BaseStockfishRequest):
        multipv: int = Field(default=1, ge=1, le=10, description="Number of principal variations to return.")
        include_pv: bool = Field(default=True, description="Include principal variation moves.")


    class PlayResponse(BaseModel):
        engine_id: int
        move: Optional[str]
        san: Optional[str] = None
        ponder: Optional[str] = None


    class AnalyseResponse(BaseModel):
        engine_id: int
        results: List[Dict[str, Any]]


    class PoolStatusResponse(BaseModel):
        pool_size: int
        started: bool
        available: int
        engines: List[Dict[str, Any]]


def _serialize_score(score: Optional[chess.engine.PovScore]) -> Optional[Dict[str, Any]]:
    if not score:
        return None
    relative = score.relative
    if isinstance(relative, chess.engine.Mate):
        return {"type": "mate", "moves": relative.moves}
    return {"type": "cp", "value": relative.score()}


def _normalize_analysis_payload(analysis: Any, include_pv: bool) -> List[Dict[str, Any]]:
    if analysis is None:
        return []

    if isinstance(analysis, dict):
        analysis = [analysis]

    payload: List[Dict[str, Any]] = []
    for info in analysis:
        entry: Dict[str, Any] = {}
        score = info.get("score")
        if score:
            entry["score"] = _serialize_score(score)
        if include_pv and "pv" in info and info["pv"]:
            entry["pv"] = [move.uci() for move in info["pv"]]
        if "depth" in info:
            entry["depth"] = info["depth"]
        if "seldepth" in info:
            entry["seldepth"] = info["seldepth"]
        payload.append(entry)
    return payload


def create_stockfish_app(
    pool_size: int = 4,
    engine_path: Optional[str] = None,
    engine_options: Optional[Dict[str, Any]] = None,
) -> "FastAPI":
    if FastAPI is None or BaseModel is None:
        raise RuntimeError(
            "FastAPI and Pydantic must be installed to create the Stockfish service. "
            "Install them via `pip install fastapi uvicorn pydantic`."
        )

    pool = StockfishEnginePool(pool_size=pool_size, engine_path=engine_path, engine_options=engine_options)
    app = FastAPI(title="Local Stockfish Service", version="1.0.0")
    app.state.engine_pool = pool

    @app.on_event("startup")
    async def _startup() -> None:
        await pool.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await pool.stop()

    @app.post("/play", response_model=PlayResponse)
    async def play(request: "PlayRequest") -> "PlayResponse":
        board = chess.Board(request.fen)
        limit = request.to_limit()
        try:
            result, engine_id = await pool.play(board, limit, ponder=request.ponder)
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Stockfish play call failed.")
            raise HTTPException(status_code=500, detail=f"Engine error: {exc}") from exc

        move = result.move.uci() if result.move else None
        san = board.san(result.move) if request.include_san and result.move else None
        ponder = result.ponder.uci() if result.ponder else None

        return PlayResponse(engine_id=engine_id, move=move, san=san, ponder=ponder)

    @app.post("/analyse", response_model=AnalyseResponse)
    async def analyse(request: "AnalyseRequest") -> "AnalyseResponse":
        board = chess.Board(request.fen)
        limit = request.to_limit()
        try:
            analysis, engine_id = await pool.analyse(board, limit, multipv=request.multipv)
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Stockfish analysis call failed.")
            raise HTTPException(status_code=500, detail=f"Engine error: {exc}") from exc

        results = _normalize_analysis_payload(analysis, include_pv=request.include_pv)
        return AnalyseResponse(engine_id=engine_id, results=results)

    @app.get("/status", response_model=PoolStatusResponse)
    async def status() -> "PoolStatusResponse":
        return PoolStatusResponse(**pool.status())

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return {"status": "ok" if pool.is_started else "starting", "available": pool.available}

    return app


def run_stockfish_service(
    host: str = "127.0.0.1",
    port: int = 8080,
    pool_size: int = 4,
    engine_path: Optional[str] = None,
    engine_options: Optional[Dict[str, Any]] = None,
    log_level: str = "info",
) -> None:
    try:
        uvicorn_module = importlib.import_module("uvicorn")
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("uvicorn must be installed to run the Stockfish service.") from exc

    app = create_stockfish_app(pool_size=pool_size, engine_path=engine_path, engine_options=engine_options)
    uvicorn_module.run(app, host=host, port=port, log_level=log_level)


async def _demo(pool_size: int, engine_path: Optional[str]) -> None:
    pool = StockfishEnginePool(pool_size=pool_size, engine_path=engine_path)
    await pool.start()

    board = chess.Board()
    limit = chess.engine.Limit(time=0.1)

    result, engine_id = await pool.play(board, limit)
    print(f"[engine {engine_id}] Suggested move: {result.move}")

    analysis, engine_id = await pool.analyse(board, chess.engine.Limit(depth=12), multipv=2)
    print(f"[engine {engine_id}] Analysis: {_normalize_analysis_payload(analysis, include_pv=True)}")

    await pool.stop()


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stockfish local service utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    service_cmd = subparsers.add_parser("serve", help="Run the Stockfish HTTP service.")
    service_cmd.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    service_cmd.add_argument("--port", type=int, default=8080, help="Port to bind.")
    service_cmd.add_argument("--pool-size", type=int, default=4, help="Number of engine workers to launch.")
    service_cmd.add_argument("--engine-path", default=None, help="Path to the Stockfish executable.")
    service_cmd.add_argument("--log-level", default="info", help="Log level for uvicorn.")
    service_cmd.add_argument(
        "--skill-level",
        type=int,
        default=None,
        help="Optional Stockfish skill level (0-20). If unset, engine defaults are used.",
    )

    demo_cmd = subparsers.add_parser("demo", help="Run a quick asynchronous demo using the engine pool.")
    demo_cmd.add_argument("--pool-size", type=int, default=2, help="Number of engine workers to launch.")
    demo_cmd.add_argument("--engine-path", default=None, help="Path to the Stockfish executable.")

    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    if args.command == "serve":
        engine_options: Optional[Dict[str, Any]] = None
        if args.skill_level is not None:
            engine_options = {"Skill Level": args.skill_level}

        run_stockfish_service(
            host=args.host,
            port=args.port,
            pool_size=args.pool_size,
            engine_path=args.engine_path,
            log_level=args.log_level,
            engine_options=engine_options,
        )
    elif args.command == "demo":
        asyncio.run(_demo(pool_size=args.pool_size, engine_path=args.engine_path))
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
