#!/usr/bin/env python3
"""
Benchmark concurrent inference server performance.

Measures:
- Single-user baseline (TTFT, total time, tokens/sec)
- N concurrent users (same metrics per user)
- Compares to validate batching benefit

Usage:
    python benchmark_concurrent.py [--url http://localhost:8080] [--users 10]
"""

import asyncio
import aiohttp
import time
import argparse
import json
from dataclasses import dataclass
from typing import Optional
import statistics


@dataclass
class RequestMetrics:
    user_id: int
    time_to_first_token: float  # seconds
    total_time: float           # seconds
    token_count: int
    tokens_per_sec: float
    response_text: str = ""     # full response for verification
    error: Optional[str] = None
    isolation_ok: Optional[bool] = None  # True if response matches expected


# Default prompt for timing-only mode
DEFAULT_PROMPT = "Write a Python function that calculates fibonacci numbers recursively. ONLY code, NO explanation!"
MAX_TOKENS = 100


def get_verification_prompt(user_id: int) -> str:
    """Generate unique prompt for isolation verification."""
    # Use a magic number derived from user_id that's unlikely to appear by chance
    magic = 7000 + user_id * 13
    return f"Write a Python function called 'get_user_value' that returns the integer {magic}. ONLY code, NO explanation!"


def verify_response(user_id: int, response_text: str) -> bool:
    """Check if response contains the expected magic number."""
    magic = 7000 + user_id * 13
    return str(magic) in response_text


async def stream_completion(
    session: aiohttp.ClientSession,
    url: str,
    user_id: int,
    prompt: str,
    max_tokens: int,
    verify: bool = False
) -> RequestMetrics:
    """Make a streaming request and measure timing."""
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    token_count = 0
    response_text = ""
    error: Optional[str] = None
    
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            if resp.status != 200:
                error = f"HTTP {resp.status}"
            else:
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        
                        # Extract token text for verification
                        try:
                            chunk = json.loads(data)
                            token_text = chunk.get("choices", [{}])[0].get("text", "")
                            response_text += token_text
                        except json.JSONDecodeError:
                            pass
                        
    except Exception as e:
        error = str(e)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else total_time
    tps = token_count / total_time if total_time > 0 else 0
    
    # Verify isolation if requested
    isolation_ok = None
    if verify and error is None:
        isolation_ok = verify_response(user_id, response_text)
    
    return RequestMetrics(
        user_id=user_id,
        time_to_first_token=ttft,
        total_time=total_time,
        token_count=token_count,
        tokens_per_sec=tps,
        response_text=response_text,
        error=error,
        isolation_ok=isolation_ok
    )


async def run_single_user(url: str, verify: bool = False) -> RequestMetrics:
    """Run single-user baseline."""
    prompt = get_verification_prompt(0) if verify else DEFAULT_PROMPT
    async with aiohttp.ClientSession() as session:
        return await stream_completion(session, url, 0, prompt, MAX_TOKENS, verify)


async def run_concurrent_users(url: str, num_users: int, verify: bool = False) -> list[RequestMetrics]:
    """Run N concurrent users."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            stream_completion(
                session, url, i,
                get_verification_prompt(i) if verify else DEFAULT_PROMPT,
                MAX_TOKENS,
                verify
            )
            for i in range(num_users)
        ]
        # Fire all at once
        return await asyncio.gather(*tasks)


def print_metrics(label: str, metrics: list[RequestMetrics]):
    """Print summary statistics."""
    successful = [m for m in metrics if m.error is None]
    failed = [m for m in metrics if m.error is not None]
    
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    
    if failed:
        print(f"  Failed: {len(failed)}/{len(metrics)}")
        for m in failed:
            print(f"    User {m.user_id}: {m.error}")
    
    if not successful:
        print("  No successful requests")
        return
    
    ttfts = [m.time_to_first_token for m in successful]
    totals = [m.total_time for m in successful]
    tokens = [m.token_count for m in successful]
    tps_list = [m.tokens_per_sec for m in successful]
    
    print(f"  Successful: {len(successful)}/{len(metrics)}")
    print()
    print(f"  Time to First Token (TTFT):")
    print(f"    Min:    {min(ttfts)*1000:7.1f} ms")
    print(f"    Max:    {max(ttfts)*1000:7.1f} ms")
    print(f"    Mean:   {statistics.mean(ttfts)*1000:7.1f} ms")
    if len(ttfts) > 1:
        print(f"    Stdev:  {statistics.stdev(ttfts)*1000:7.1f} ms")
    print()
    print(f"  Total Completion Time:")
    print(f"    Min:    {min(totals):7.2f} s")
    print(f"    Max:    {max(totals):7.2f} s")
    print(f"    Mean:   {statistics.mean(totals):7.2f} s")
    print()
    print(f"  Tokens Generated:")
    print(f"    Total:  {sum(tokens)}")
    print(f"    Mean:   {statistics.mean(tokens):.1f} per request")
    print()
    print(f"  Throughput (per user):")
    print(f"    Mean:   {statistics.mean(tps_list):7.2f} tok/s")
    print()
    
    # Wall-clock throughput (total tokens / max completion time)
    wall_time = max(totals)
    total_tokens = sum(tokens)
    print(f"  Aggregate Throughput:")
    print(f"    Wall time:     {wall_time:.2f} s")
    print(f"    Total tokens:  {total_tokens}")
    print(f"    Throughput:    {total_tokens/wall_time:.2f} tok/s")


def print_comparison(single: RequestMetrics, concurrent: list[RequestMetrics]):
    """Print comparison between single and concurrent."""
    successful = [m for m in concurrent if m.error is None]
    if not successful or single.error:
        return
    
    print(f"\n{'='*60}")
    print(f" COMPARISON: Single vs {len(concurrent)} Concurrent")
    print(f"{'='*60}")
    
    # Single user metrics
    single_tps = single.tokens_per_sec
    single_total = single.total_time
    
    # Concurrent metrics
    conc_totals = [m.total_time for m in successful]
    conc_max_total = max(conc_totals)
    conc_mean_total = statistics.mean(conc_totals)
    conc_total_tokens = sum(m.token_count for m in successful)
    conc_throughput = conc_total_tokens / conc_max_total
    
    print()
    print(f"  Single user:")
    print(f"    Total time:    {single_total:.2f} s")
    print(f"    Throughput:    {single_tps:.2f} tok/s")
    print()
    print(f"  {len(successful)} concurrent users:")
    print(f"    Mean time:     {conc_mean_total:.2f} s  ({conc_mean_total/single_total:.2f}x single)")
    print(f"    Max time:      {conc_max_total:.2f} s  ({conc_max_total/single_total:.2f}x single)")
    print(f"    Throughput:    {conc_throughput:.2f} tok/s  ({conc_throughput/single_tps:.2f}x single)")
    print()
    
    # Verdict
    slowdown = conc_max_total / single_total
    speedup = conc_throughput / single_tps
    
    if slowdown < 2.0 and speedup > 3.0:
        verdict = "✓ EXCELLENT - Batching working well"
    elif slowdown < 3.0 and speedup > 2.0:
        verdict = "✓ GOOD - Batching providing benefit"
    elif speedup > 1.5:
        verdict = "~ OK - Some batching benefit"
    else:
        verdict = "✗ POOR - Batching may not be working"
    
    print(f"  Verdict: {verdict}")
    print()


def print_isolation_results(metrics: list[RequestMetrics]):
    """Print isolation verification results."""
    verified = [m for m in metrics if m.isolation_ok is not None]
    if not verified:
        return
    
    passed = [m for m in verified if m.isolation_ok]
    failed = [m for m in verified if not m.isolation_ok]
    
    print(f"\n{'='*60}")
    print(f" ISOLATION VERIFICATION")
    print(f"{'='*60}")
    print()
    print(f"  Passed: {len(passed)}/{len(verified)}")
    print(f"  Failed: {len(failed)}/{len(verified)}")
    
    if failed:
        print()
        print(f"  FAILURES (user got wrong response):")
        for m in failed:
            expected_magic = 7000 + m.user_id * 13
            print(f"    User {m.user_id}: expected {expected_magic}")
            # Show snippet of what they got
            snippet = m.response_text[:100].replace('\n', '\\n')
            print(f"      Got: {snippet}...")
    
    print()
    if len(failed) == 0:
        print(f"  ✓ ALL USERS RECEIVED CORRECT ISOLATED RESPONSES")
    else:
        print(f"  ✗ ISOLATION FAILURE - Responses leaked between slots!")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Benchmark concurrent inference")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--skip-single", action="store_true", help="Skip single-user baseline")
    parser.add_argument("--verify", action="store_true", help="Verify isolation (each user gets unique prompt)")
    args = parser.parse_args()
    
    print(f"Server: {args.url}")
    if args.verify:
        print(f"Mode: VERIFICATION (each user gets unique prompt)")
        print(f"Prompt pattern: 'return {7000 + 0*13}', 'return {7000 + 1*13}', ...")
    else:
        print(f"Prompt: {DEFAULT_PROMPT[:50]}...")
    print(f"Max tokens: {MAX_TOKENS}")
    
    # Single user baseline
    single_metrics = None
    if not args.skip_single:
        print("\nRunning single-user baseline...")
        single_metrics = await run_single_user(args.url, args.verify)
        print_metrics("Single User Baseline", [single_metrics])
    
    # Concurrent users
    print(f"\nRunning {args.users} concurrent users...")
    concurrent_metrics = await run_concurrent_users(args.url, args.users, args.verify)
    print_metrics(f"{args.users} Concurrent Users", concurrent_metrics)
    
    # Comparison
    if single_metrics and not single_metrics.error:
        print_comparison(single_metrics, concurrent_metrics)
    
    # Isolation verification
    if args.verify:
        print_isolation_results(concurrent_metrics)
    
    # Per-user breakdown
    print(f"\n{'='*60}")
    print(f" Per-User Breakdown")
    print(f"{'='*60}")
    if args.verify:
        print(f"  {'User':>4}  {'TTFT':>8}  {'Total':>8}  {'Tokens':>6}  {'tok/s':>7}  {'Isolation':>9}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}")
    else:
        print(f"  {'User':>4}  {'TTFT':>8}  {'Total':>8}  {'Tokens':>6}  {'tok/s':>7}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*7}")
    for m in sorted(concurrent_metrics, key=lambda x: x.time_to_first_token):
        if m.error:
            print(f"  {m.user_id:>4}  ERROR: {m.error}")
        else:
            iso_status = ""
            if args.verify:
                iso_status = "  ✓ PASS" if m.isolation_ok else "  ✗ FAIL"
            print(f"  {m.user_id:>4}  {m.time_to_first_token*1000:>7.0f}ms  {m.total_time:>7.2f}s  {m.token_count:>6}  {m.tokens_per_sec:>7.2f}{iso_status}")


if __name__ == "__main__":
    asyncio.run(main())
