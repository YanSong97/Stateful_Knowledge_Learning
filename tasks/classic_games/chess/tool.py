import asyncio
import importlib
import os
import shutil
from typing import Any, Dict, List, Optional

try:
    chess = importlib.import_module("chess")
    importlib.import_module("chess.engine")
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("python-chess must be installed to use StockFish_Tool.") from exc

try:
    httpx = importlib.import_module("httpx")
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("The httpx package must be installed to use StockFish_Client.") from exc


import requests

# from tasks.classic_games.chess.chess_env import ChessEnv_Wrapper

# https://python-chess.readthedocs.io/en/latest/engine.html
# https://blog.propelauth.com/chess-analysis-in-python/

class StockFish_Tool:
    def __init__(self, engine_path: Optional[str] = None):
        self.engine_path = engine_path
        self.engine = None

    def _resolve_engine_path(self, engine_path: Optional[str] = None) -> str:
        candidates = [
            engine_path,
            self.engine_path,
            os.environ.get("STOCKFISH_EXECUTABLE"),
            shutil.which("stockfish"),
        ]

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        raise FileNotFoundError(
            "Stockfish executable path not found. Pass engine_path, set "
            "STOCKFISH_EXECUTABLE, or install stockfish on PATH."
        )

    def load_engine(self, engine_path: Optional[str] = None):
        stockfish_engine_path = self._resolve_engine_path(engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_engine_path)
        # print("Stockfish engine loaded successfully.")


    def infer_action(self, board: chess.Board, n: int):
        if self.engine is None:
            raise RuntimeError("Please initialize the backend engine first")
        if n == 1:
            stockfish_result = self.engine.play(board, chess.engine.Limit(time=0.1))
            move = stockfish_result.move
            return move
        elif n > 1:
            analysis_result = self.engine.analysis(board, limit=chess.engine.Limit(depth=18), multipv=n)
            # analysis_result.wait()
            analysed_variations = analysis_result.multipv
            return analysed_variations
        else:
            raise NotImplementedError

    def infer_state_score(self, board):
        if self.engine is None:
            raise RuntimeError("Please initialize the backend engine first")

        stockfish_info = self.engine.analyse(board, chess.engine.Limit(depth=20))
        # print(f"STOCKFISH info = {stockfish_info['score']}")
        # print(f"STOCKFISH = {stockfish_info['score'].relative.score()}")

        if isinstance(stockfish_info['score'].relative, chess.engine.Mate):
            return 2000
        else:
            return stockfish_info['score'].relative.score()


    def infer_analysis(self, board):
        if self.engine is None:
            raise RuntimeError("Please initialize the backend engine first")

        stockfish_info = self.engine.analyse(board, chess.engine.Limit(depth=20))
        return stockfish_info

    def quit(self):
        if self.engine:
            self.engine.quit()
            print("Stockfish engine has been shut down.")


class StockFish_Client:
    """Async HTTP client for interacting with a running Stockfish service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "StockFish_Client":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def play_move(
        self,
        fen: str,
        *,
        time_limit: Optional[float] = 0.1,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        include_san: bool = False,
        ponder: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"fen": fen, "include_san": include_san, "ponder": ponder}
        if time_limit is not None:
            payload["time_limit"] = time_limit
        if depth is not None:
            payload["depth"] = depth
        if nodes is not None:
            payload["nodes"] = nodes

        client = await self._ensure_client()
        response = await client.post("/play", json=payload)
        response.raise_for_status()
        return response.json()

    async def analyse_position(
        self,
        fen: str,
        *,
        time_limit: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        multipv: int = 1,
        include_pv: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"fen": fen, "multipv": multipv, "include_pv": include_pv}
        if time_limit is not None:
            payload["time_limit"] = time_limit
        if depth is not None:
            payload["depth"] = depth
        if nodes is not None:
            payload["nodes"] = nodes

        client = await self._ensure_client()
        response = await client.post("/analyse", json=payload)
        response.raise_for_status()
        return response.json()

    async def status(self) -> Dict[str, Any]:
        client = await self._ensure_client()
        response = await client.get("/status")
        response.raise_for_status()
        return response.json()


class StockFish_http_client:
    """Asynchronous HTTP client for interacting with a running Stockfish service using requests."""

    def __init__(self, search_url: str, timeout: float = 30.0) -> None:
        self.search_url = search_url
        self.timeout = timeout

    async def play_move(
        self,
        fen: str,
        *,
        time_limit: Optional[float] = 0.1,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        include_san: bool = False,
        ponder: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"fen": fen, "include_san": include_san, "ponder": ponder}
        if time_limit is not None:
            payload["time_limit"] = time_limit
        if depth is not None:
            payload["depth"] = depth
        if nodes is not None:
            payload["nodes"] = nodes

        response = requests.post(self.search_url, json=payload, timeout=self.timeout).json()
        return response

    async def analyse_position(
        self,
        fen: str,
        *,
        time_limit: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        multipv: int = 1,
        include_pv: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"fen": fen, "multipv": multipv, "include_pv": include_pv}
        if time_limit is not None:
            payload["time_limit"] = time_limit
        if depth is not None:
            payload["depth"] = depth
        if nodes is not None:
            payload["nodes"] = nodes

        response = requests.post(self.search_url, json=payload, timeout=self.timeout).json()

        # post-process
        # response = {'engine_id': 15, 'results': [{'score': {'type': 'cp', 'value': 4}, 'pv': ['e2f3', 'e1d2', 'f3e4', 'd2c1'], 'depth': 8, 'seldepth': 11}, {'score': {'type': 'cp', 'value': 2}, 'pv': ['e2e3', 'e1f1', 'e3e4', 'f1e2'], 'depth': 8, 'seldepth': 9}]}

        return response

    async def status(self) -> Dict[str, Any]:
        response = requests.get(self.search_url, timeout=self.timeout).json()
        return response


class ValueTable_http_client:
    """Asynchronous HTTP client for interacting with a running value table service using requests."""

    def __init__(self, base_url: str = "http://127.0.0.1:8081", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def store(self, state: str, value: str) -> Dict[str, Any]:
        """Store a value for a state. Multiple values can be stored for the same state.
        
        Args:
            state: FEN string representing the board state
            value: String describing the state
            
        Returns:
            Response dict with 'success', 'is_new', 'value_count', and 'message' keys
        """
        payload: Dict[str, Any] = {"state": state, "value": value}
        response = requests.post(
            f"{self.base_url}/store", json=payload, timeout=self.timeout
        ).json()
        return response

    async def get(self, state: str, retrieve_strategy: str = "random") -> Dict[str, Any]:
        """Retrieve a value for a given state using the specified strategy.
        
        Args:
            state: FEN string representing the board state
            retrieve_strategy: Strategy to use - "random" (randomly sample) or "latest" (get latest/highest order)
            
        Returns:
            Response dict with 'found', 'value', 'order', 'value_count', and 'message' keys
        """
        payload: Dict[str, Any] = {"state": state, "retrieve_strategy": retrieve_strategy}
        response = requests.post(
            f"{self.base_url}/get", json=payload, timeout=self.timeout
        ).json()
        return response

    async def batch_store(self, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store multiple (state, value) pairs in a single operation.
        Multiple values can be stored for the same state.
        
        Args:
            pairs: List of dicts with 'state' and 'value' keys
            
        Returns:
            Response dict with 'stored', 'added', 'errors', and 'total' keys
        """
        payload: Dict[str, Any] = {"pairs": pairs}
        response = requests.post(
            f"{self.base_url}/batch_store", json=payload, timeout=self.timeout
        ).json()
        return response

    async def batch_get(self, states: List[str], retrieve_strategy: str = "random") -> Dict[str, Any]:
        """Retrieve values for multiple states using the specified strategy.
        
        Args:
            states: List of FEN strings to query
            retrieve_strategy: Strategy to use - "random" (randomly sample) or "latest" (get latest/highest order)
            
        Returns:
            Response dict with 'results' key containing a dict mapping states to value info
            Each value info contains 'value', 'order', and 'value_count'
        """
        payload: Dict[str, Any] = {"states": states, "retrieve_strategy": retrieve_strategy}
        response = requests.post(
            f"{self.base_url}/batch_get", json=payload, timeout=self.timeout
        ).json()
        return response

    async def clear(self) -> Dict[str, Any]:
        """Clear all entries from the value table.
        
        Returns:
            Response dict with 'success', 'cleared', and 'message' keys
        """
        response = requests.delete(f"{self.base_url}/clear", timeout=self.timeout).json()
        return response

    async def status(self) -> Dict[str, Any]:
        """Get status information about the value table.
        
        Returns:
            Response dict with 'size', 'total_instances', 'total_stores', 'total_retrieves', 'uptime_seconds', and 'created_at' keys
        """
        response = requests.get(f"{self.base_url}/status", timeout=self.timeout).json()
        return response

    async def check(self) -> Dict[str, Any]:
        """Check the value table: list covered states and count instances.
        
        Returns:
            Response dict with 'total_states', 'total_instances', and 'states' (list of states with counts)
        """
        response = requests.get(f"{self.base_url}/check", timeout=self.timeout).json()
        return response

    async def healthz(self) -> Dict[str, Any]:
        """Health check endpoint.
        
        Returns:
            Response dict with 'status', 'size', and 'total_instances' keys
        """
        response = requests.get(f"{self.base_url}/healthz", timeout=self.timeout).json()
        return response


if __name__ == "__main__":
    async def remote_client_test() -> None:
        print("\n--- Testing StockFish_Client (remote service) ---")
        test_fens = [
            chess.STARTING_FEN,
            "rn1qkbnr/ppp2ppp/4p3/3p4/3P4/5NP1/PPP1PPBP/RNBQK2R w KQkq - 0 5",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ]
        base_url = os.environ.get("STOCKFISH_URL")
        if not base_url:
            raise ValueError("Set STOCKFISH_URL to run the remote Stockfish client test.")
        client = StockFish_Client(base_url=base_url)
        try:
            async with client:
                status_info = await client.status()
                # print(f"Service status: {status_info}")

                play_tasks = [
                    asyncio.create_task(client.play_move(fen, time_limit=0.1, include_san=True))
                    for fen in test_fens
                ]
                play_results = await asyncio.gather(*play_tasks, return_exceptions=True)
                for idx, result in enumerate(play_results):
                    if isinstance(result, Exception):
                        print(f"Play request {idx} failed: {result}")
                    else:
                        print(f"Play request {idx} result: {result}")

                analyse_tasks = [
                    asyncio.create_task(client.analyse_position(fen, depth=8, multipv=2))
                    for fen in test_fens
                ]
                analyse_results = await asyncio.gather(*analyse_tasks, return_exceptions=True)
                for idx, result in enumerate(analyse_results):
                    if isinstance(result, Exception):
                        print(f"Analyse request {idx} failed: {result}")
                    else:
                        print(f"Analyse request {idx} result: {result}")
        except Exception as exc:  # noqa: BLE001 - demo output
            print(f"Skipping remote client test: {exc}")

    async def value_table_client_test() -> None:
        """Test suite for ValueTable_http_client."""
        print("\n" + "="*60)
        print("--- Testing ValueTable_http_client (remote service) ---")
        print("="*60)

        # Test FEN strings
        test_fens = [
            chess.STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "rn1qkbnr/ppp2ppp/4p3/3p4/3P4/5NP1/PPP1PPBP/RNBQK2R w KQkq - 0 5",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        ]

        # Default base URL, can be overridden via environment variable
        base_url = os.environ.get("VALUE_TABLE_URL", "http://127.0.0.1:8081")
        client = ValueTable_http_client(base_url=base_url, timeout=10.0)

        try:
            # Test 1: Health check
            print("\n--- Test 1: Health Check ---")
            try:
                health = await client.healthz()
                print(f"✓ Health check passed: {health}")
            except Exception as exc:
                print(f"✗ Health check failed: {exc}")
                print("Make sure the value table server is running!")
                return

            # Test 2: Store single values
            print("\n--- Test 2: Store Single Values ---")
            store_tasks = []
            test_values = [0.5, 0.7, -0.3, 0.2]
            for fen, value in zip(test_fens, test_values):
                task = asyncio.create_task(client.store(fen, value))
                store_tasks.append((fen, value, task))

            for fen, expected_value, task in store_tasks:
                try:
                    result = await task
                    if result.get("success"):
                        is_new = result.get("is_new", False)
                        print(f"✓ Stored {fen[:30]}... (value: {expected_value}, new: {is_new})")
                    else:
                        print(f"✗ Failed to store {fen[:30]}...: {result}")
                except Exception as exc:
                    print(f"✗ Store failed for {fen[:30]}...: {exc}")

            # Test 3: Get single values
            print("\n--- Test 3: Get Single Values ---")
            get_tasks = [
                asyncio.create_task(client.get(fen))
                for fen in test_fens
            ]
            get_results = await asyncio.gather(*get_tasks, return_exceptions=True)

            for idx, (fen, expected_value) in enumerate(zip(test_fens, test_values)):
                result = get_results[idx]
                if isinstance(result, Exception):
                    print(f"✗ Get request {idx} failed: {result}")
                else:
                    found = result.get("found", False)
                    value = result.get("value")
                    if found and value == expected_value:
                        print(f"✓ Retrieved {fen[:30]}... (value: {value})")
                    else:
                        print(f"✗ Get mismatch for {fen[:30]}... (found: {found}, value: {value}, expected: {expected_value})")

            # Test 4: Batch store
            print("\n--- Test 4: Batch Store ---")
            batch_pairs = [
                {"state": test_fens[0], "value": 0.6},  # Update existing
                {"state": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "value": 0.4},  # New
                {"state": test_fens[2], "value": -0.5},  # Update existing
            ]
            try:
                batch_result = await client.batch_store(batch_pairs)
                print("✓ Batch store completed:")
                print(f"  - Stored: {batch_result.get('stored', 0)}")
                print(f"  - Updated: {batch_result.get('updated', 0)}")
                print(f"  - Errors: {batch_result.get('errors', 0)}")
                print(f"  - Total: {batch_result.get('total', 0)}")
            except Exception as exc:
                print(f"✗ Batch store failed: {exc}")

            # Test 5: Batch get
            print("\n--- Test 5: Batch Get ---")
            batch_states = test_fens + ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"]
            try:
                batch_get_result = await client.batch_get(batch_states)
                results = batch_get_result.get("results", {})
                print(f"✓ Batch get completed for {len(batch_states)} states:")
                for state in batch_states:
                    value = results.get(state)
                    if value is not None:
                        print(f"  - {state[:40]}... -> {value}")
                    else:
                        print(f"  - {state[:40]}... -> Not found")
            except Exception as exc:
                print(f"✗ Batch get failed: {exc}")

            # Test 6: Status
            print("\n--- Test 6: Status Check ---")
            try:
                status = await client.status()
                print("✓ Status retrieved:")
                print(f"  - Size: {status.get('size', 0)}")
                print(f"  - Total stores: {status.get('total_stores', 0)}")
                print(f"  - Total retrieves: {status.get('total_retrieves', 0)}")
                print(f"  - Uptime: {status.get('uptime_seconds', 0):.2f} seconds")
            except Exception as exc:
                print(f"✗ Status check failed: {exc}")

            # Test 7: Concurrent operations
            print("\n--- Test 7: Concurrent Operations ---")
            concurrent_tasks = []
            for i, fen in enumerate(test_fens):
                # Mix of store and get operations
                if i % 2 == 0:
                    concurrent_tasks.append(
                        asyncio.create_task(client.store(fen, 0.1 * i))
                    )
                else:
                    concurrent_tasks.append(
                        asyncio.create_task(client.get(fen))
                    )

            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            success_count = sum(1 for r in concurrent_results if not isinstance(r, Exception))
            print(f"✓ Concurrent operations: {success_count}/{len(concurrent_tasks)} succeeded")

            # Test 8: Clear (optional - comment out if you want to keep data)
            print("\n--- Test 8: Clear Table (optional) ---")
            try:
                clear_result = await client.clear()
                cleared = clear_result.get("cleared", 0)
                print(f"✓ Cleared {cleared} entries from the table")
            except Exception as exc:
                print(f"✗ Clear failed: {exc}")

            print("\n" + "="*60)
            print("--- ValueTable_http_client Test Complete ---")
            print("="*60 + "\n")

        except Exception as exc:  # noqa: BLE001 - demo output
            print(f"\n✗ Test suite failed with error: {exc}")
            import traceback
            traceback.print_exc()

    # Run tests
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "value_table":
        asyncio.run(value_table_client_test())
    else:
        asyncio.run(remote_client_test())


# if False:
#     async def main_test():
#         """
#         A test suite to demonstrate and verify the functionality of the StockFish_Tool.
#         """
#         print("--- Starting StockFish_Tool Test ---")

#         # 1. Initialize the tool and load the engine
#         tool = StockFish_Tool()
#         tool.load_engine()

#         # Abort if the engine failed to load
#         if not tool.engine:
#             print("Test aborted: Engine could not be loaded.")
#             return

#         # 2. Set up a chess environment
#         game = ChessEnv_Wrapper(None)
#         game.reset()
#         print("\nInitial board state:")
#         game.render()

#         # 3. Test infer_action for a single best move (n=1)
#         print("\n--- Testing infer_action(n=1) ---")
#         best_move = tool.infer_action(game.game, n=1)
#         if best_move:
#             print(f"Stockfish suggests the best move is: {best_move}")
#         else:
#             print("Could not determine the best move.")

#         # # 4. Test infer_action for the top 3 moves (n=3)
#         print("\n--- Testing infer_action(n=3) ---")
#         top_moves_analysis = tool.infer_action(game.game, n=3)
#         if top_moves_analysis:
#             print("Top 3 moves analysis:")
#             for i, info in enumerate(top_moves_analysis):
#                 # The 'pv' (principal variation) is a list of moves, the first is the one being scored
#                 move = info.get('pv', [None])[0]
#                 score = info.get('score')
#                 print(f"  Rank {i + 1}: Move {move}, Score: {score}")
#         else:
#             print("Could not get top moves analysis.")
#         #
#         # 5. Test infer_analysis for a deep evaluation
#         print("\n--- Testing infer_analysis (deep evaluation) ---")
#         deep_analysis = tool.infer_analysis(game.game)

#         if deep_analysis:
#             print(f"Deep analysis score: {deep_analysis.get('score').relative.score()}")
#         else:
#             print("Could not perform deep analysis.")

#         # 6. infer state score
#         state_score = tool.infer_state_score(game.game)
#         print(f"state score = {state_score}")

#         # 6. Clean up by quitting the engine
#         print("\n--- Shutting down engine ---")
#         tool.quit()
#         print("\n--- Test Complete ---")

#     # Run the asynchronous test function
#     asyncio.run(main_test())
