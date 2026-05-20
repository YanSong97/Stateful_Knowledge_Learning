import argparse
import asyncio
import json
import random
import statistics
import time
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError as exc:  # pragma: no cover - dependency check
    raise RuntimeError("Install httpx to run the Stockfish benchmark script (`pip install httpx`).") from exc


_DEFAULT_FENS: List[str] = [
    "8/2nk2b1/p2p1npp/N1pPp3/2P1P2P/P1N1B1P1/7K/8 w - - 5 34",
    "2r5/2p5/p1Pp3k/1p5B/1P3q1P/P3bQK1/2R5/8 w - - 1 37",
    "rn3rk1/pbpp2pp/1p2pnq1/5p2/2PP4/P1P1PN2/2Q1BPPP/R1B2RK1 w - - 7 11",
    "r1bqkb1Q/pp5p/4p1p1/1pn5/8/2N5/PPK2PPP/R1B3NR w q - 0 14",
]


async def _worker(
    client: httpx.AsyncClient,
    endpoint: str,
    payloads: List[Dict[str, Any]],
    samples: List[float],
    semaphore: asyncio.Semaphore,
    worker_id: int,
    timeout: float,
    quiet: bool,
) -> None:
    while True:
        async with semaphore:
            if not payloads:
                return
            payload = payloads.pop()

        start = time.perf_counter()
        try:
            resp = await client.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - capture failures
            if not quiet:
                print(f"[worker {worker_id}] request failed: {exc}")
            continue
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            samples.append(elapsed_ms)

        if not quiet:
            print(f"[worker {worker_id}] {elapsed_ms:.2f} ms :: {json.dumps(data, ensure_ascii=False)}")


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _build_payloads(
    total_requests: int,
    fen_list: List[str],
    depth: Optional[int],
    time_limit: Optional[float],
    multipv: int,
    include_san: bool,
    include_pv: bool,
    endpoint_choice: str,
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for _ in range(total_requests):
        fen = random.choice(fen_list)
        base: Dict[str, Any] = {"fen": fen}
        if depth is not None:
            base["depth"] = depth
        if time_limit is not None:
            base["time_limit"] = time_limit

        if endpoint_choice == "play":
            base["include_san"] = include_san
        else:
            base["multipv"] = multipv
            base["include_pv"] = include_pv

        payloads.append(base)
    return payloads


async def run_benchmark(args: argparse.Namespace) -> None:
    endpoint = f"{args.base_url.rstrip('/')}/{args.endpoint}"
    payloads = _build_payloads(
        total_requests=args.requests,
        fen_list=args.fens,
        depth=args.depth,
        time_limit=args.time,
        multipv=args.multipv,
        include_san=args.include_san,
        include_pv=args.include_pv,
        endpoint_choice=args.endpoint,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    samples: List[float] = []

    async with httpx.AsyncClient() as client:
        workers = [
            _worker(client, endpoint, payloads, samples, semaphore, worker_id=i, timeout=args.timeout, quiet=args.quiet)
            for i in range(args.concurrency)
        ]
        start = time.perf_counter()
        await asyncio.gather(*workers)
        total_time = time.perf_counter() - start

    if not samples:
        print("No successful samples collected.")
        return

    mean_ms = statistics.fmean(samples)
    median_ms = statistics.median(samples)
    p90 = _percentile(samples, 0.90)
    p95 = _percentile(samples, 0.95)
    p99 = _percentile(samples, 0.99)
    rps = len(samples) / total_time if total_time > 0 else 0.0

    print("\n=== Stockfish Service Benchmark ===")
    print(f"Endpoint        : {endpoint}")
    print(f"Requests        : {len(samples)} / {args.requests} attempted")
    print(f"Concurrency     : {args.concurrency}")
    print(f"Total time      : {total_time:.2f} s")
    print(f"Throughput      : {rps:.2f} req/s")
    print(f"Mean latency    : {mean_ms:.2f} ms")
    print(f"Median latency  : {median_ms:.2f} ms")
    print(f"P90 latency     : {p90:.2f} ms")
    print(f"P95 latency     : {p95:.2f} ms")
    print(f"P99 latency     : {p99:.2f} ms")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Stockfish HTTP service.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Service base URL.")
    parser.add_argument("--endpoint", choices=["play", "analyse"], default="play", help="Endpoint to test.")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests to issue.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP client timeout per request (seconds).")
    parser.add_argument("--depth", type=int, default=None, help="Depth limit to pass to the service (optional).")
    parser.add_argument("--time", type=float, default=0.1, help="Time limit (seconds) to pass to the service.")
    parser.add_argument("--multipv", type=int, default=1, help="PV count for analyse requests.")
    parser.add_argument("--include-san", action="store_true", help="Ask /play for SAN notation in responses.")
    parser.add_argument("--include-pv", action="store_true", help="Ask /analyse to include principal variation moves.")
    parser.add_argument(
        "--fen",
        action="append",
        dest="fens",
        default=None,
        help="Provide explicit FENs (can be repeated). Defaults to a built-in selection.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-request logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.fens:
        args.fens = list(_DEFAULT_FENS)
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()

    # python tasks/classic_games/chess/stockfish_bench.py --endpoint analyse --base-url "$STOCKFISH_URL"
