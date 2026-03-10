#!/usr/bin/env python3
"""
Benchmark Order Management DSL with system prompt caching.

Tests prefix caching by sending 9 concurrent requests that share
the same system prompt but have different user prompts.

Usage:
    python benchmark_order_management.py [--url http://localhost:8080]
"""

import asyncio
import aiohttp
import time
import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional
import statistics


SYSTEM_PROMPT = """You are a code generator for an Order Management System.
You output ONLY valid JavaScript. No explanation, no comments, no markdown.

=== ENTITIES ===

Customer: { id, name, tier, region, status }
  - tier: "standard" | "gold" | "platinum"
  - status: "active" | "suspended"

Order: { id, customerId, status, total, itemCount, createdAt }
  - status: "pending" | "confirmed" | "shipped" | "delivered" | "cancelled"

LineItem: { id, orderId, productId, sku, quantity, unitPrice }

Product: { id, sku, name, category, stock, price }

Shipment: { id, orderId, carrier, status, trackingNumber }
  - status: "preparing" | "in_transit" | "delivered" | "failed"

Payment: { id, orderId, status, amount, method }
  - status: "pending" | "completed" | "failed" | "refunded"

=== QUERY FUNCTIONS ===

getCustomers({ tier?, region?, status? }) → Customer[]
getOrders({ customerId?, status?, minTotal?, maxTotal? }) → Order[]
getLineItems({ orderId }) → LineItem[]
getProduct({ productId }) → Product
getProducts({ category?, inStock? }) → Product[]
getShipment({ orderId }) → Shipment
getPayments({ orderId?, status? }) → Payment[]

=== ACTION FUNCTIONS ===

updateOrderStatus({ orderId, status }) → void
cancelOrder({ orderId, reason }) → void
addOrderNote({ orderId, note }) → void
flagOrder({ orderId, reason }) → void

createShipment({ orderId, carrier }) → Shipment
updateShipment({ shipmentId, status, trackingNumber? }) → void

processPayment({ orderId, amount, method }) → Payment
refundPayment({ paymentId, reason }) → void

updateStock({ productId, quantity }) → void
assignToRep({ orderId, repId }) → void

=== ALLOWED SYNTAX ===

Variable declaration:
  const x = value;
  const x = await fn({ });

Loops:
  for (const x of array) { }

Conditionals:
  if (condition) { }
  if (condition) { } else { }
  if (condition) { } else if (condition) { } else { }

Parallel execution:
  await Promise.all(array.map(async (x) => { }));

Array operations:
  array.length
  array.filter(x => condition)
  array.map(x => value)
  array.find(x => condition)

Comparisons:
  ===  !==  >  <  >=  <=

Logic:
  &&  ||  !

=== RULES ===

1. Always use await with function calls
2. Always use const for variables
3. Function arguments must be strings, numbers, or variables
4. No console.log
5. No comments
6. No try/catch
7. No let or var
8. No string concatenation or templates
9. No arithmetic except comparisons
10. Output only code, nothing else

=== EXAMPLES ===

User: get all gold tier customers
Output:
const customers = await getCustomers({ tier: "gold" });

User: cancel all pending orders for suspended customers
Output:
const customers = await getCustomers({ status: "suspended" });
for (const customer of customers) {
  const orders = await getOrders({ customerId: customer.id, status: "pending" });
  for (const order of orders) {
    await cancelOrder({ orderId: order.id, reason: "Customer suspended" });
  }
}
"""


@dataclass 
class TestCase:
    name: str
    prompt: str
    expected_functions: list[str]
    required_constructs: list[str]
    max_tokens: int = 512


TEST_CASES = [
    TestCase(
        name="01_simple_query",
        prompt="Get all products in the electronics category",
        expected_functions=["getProducts"],
        required_constructs=["await"],
    ),
    TestCase(
        name="02_loop_with_action",
        prompt="Update all orders with status 'shipped' to 'delivered'",
        expected_functions=["getOrders", "updateOrderStatus"],
        required_constructs=["await", "for"],
    ),
    TestCase(
        name="03_nested_loop",
        prompt="For each platinum customer, get their pending orders and add a note saying 'Priority customer'",
        expected_functions=["getCustomers", "getOrders", "addOrderNote"],
        required_constructs=["await", "for"],
    ),
    TestCase(
        name="04_conditional",
        prompt="Find orders over $500 with failed payments. Refund the payment and cancel the order with reason 'Payment failed'",
        expected_functions=["getOrders", "getPayments", "refundPayment", "cancelOrder"],
        required_constructs=["await", "for", "if"],
    ),
    TestCase(
        name="05_nested_condition",
        prompt="For each order in 'confirmed' status, check if all products in its line items are in stock. If any product has stock < quantity ordered, flag the order with reason 'Insufficient stock'",
        expected_functions=["getOrders", "getLineItems", "getProduct", "flagOrder"],
        required_constructs=["await", "for", "if"],
        max_tokens=768,
    ),
    TestCase(
        name="06_aggregation",
        prompt="For customers in the EU region, get all completed orders and calculate total revenue. If total > $10000, flag all pending orders for that customer with reason 'High-value customer'",
        expected_functions=["getCustomers", "getOrders", "flagOrder"],
        required_constructs=["await", "for", "if"],
        max_tokens=768,
    ),
    TestCase(
        name="07_parallel",
        prompt="For the given order, update stock for all line items in parallel using Promise.all",
        expected_functions=["getLineItems", "updateStock"],
        required_constructs=["await", "Promise.all"],
    ),
    TestCase(
        name="08_null_handling",
        prompt="For each confirmed order, get its shipment. If no shipment exists, create one with carrier 'default'. Then update order status to 'shipped'",
        expected_functions=["getOrders", "getShipment", "createShipment", "updateOrderStatus"],
        required_constructs=["await", "for", "if"],
    ),
    TestCase(
        name="09_multi_filter",
        prompt="Find all gold tier customers in EU region with pending orders over $200. For each matching order, process payment with method 'express', then update order to confirmed, then create shipment with carrier 'premium'",
        expected_functions=["getCustomers", "getOrders", "processPayment", "updateOrderStatus", "createShipment"],
        required_constructs=["await", "for"],
    ),
]


@dataclass
class RequestResult:
    test_case: TestCase
    response_text: str
    time_to_first_token: float
    total_time: float
    token_count: int
    functions_found: list[str]
    constructs_found: list[str]
    passed: bool
    failures: list[str]
    error: Optional[str] = None


def validate_response(tc: TestCase, response: str) -> tuple[bool, list[str], list[str], list[str]]:
    """Validate generated code against test case requirements."""
    failures = []
    functions_found = []
    constructs_found = []
    
    # Check for expected functions
    for fn in tc.expected_functions:
        pattern = fn + r"\s*\("
        if re.search(pattern, response):
            functions_found.append(fn)
        else:
            failures.append(f"Missing function: {fn}")
    
    # Check for required constructs
    for construct in tc.required_constructs:
        if construct == "for":
            found = "for (" in response or "for(" in response
        elif construct == "if":
            found = "if (" in response or "if(" in response
        elif construct == "await":
            found = "await " in response
        elif construct == "Promise.all":
            found = "Promise.all" in response
        else:
            found = construct in response
        
        if found:
            constructs_found.append(construct)
        else:
            failures.append(f"Missing construct: {construct}")
    
    passed = len(failures) == 0
    return passed, failures, functions_found, constructs_found


async def run_request(
    session: aiohttp.ClientSession,
    url: str,
    tc: TestCase,
    use_system_prompt: bool = True
) -> RequestResult:
    """Run a single test case request."""
    
    payload = {
        "prompt": tc.prompt,
        "max_tokens": tc.max_tokens,
        "stream": True
    }
    if use_system_prompt:
        payload["system_prompt"] = SYSTEM_PROMPT
    else:
        # No caching: include system prompt in main prompt
        payload["prompt"] = SYSTEM_PROMPT + "\n\nUser: " + tc.prompt + "\nOutput:\n"
    
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
    
    # Validate response
    passed, failures, functions_found, constructs_found = validate_response(tc, response_text)
    if error:
        passed = False
        failures.append(f"Error: {error}")
    
    return RequestResult(
        test_case=tc,
        response_text=response_text,
        time_to_first_token=ttft,
        total_time=total_time,
        token_count=token_count,
        functions_found=functions_found,
        constructs_found=constructs_found,
        passed=passed,
        failures=failures,
        error=error
    )


async def run_all_concurrent(url: str, use_system_prompt: bool = True) -> list[RequestResult]:
    """Run all test cases concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            run_request(session, url, tc, use_system_prompt)
            for tc in TEST_CASES
        ]
        return await asyncio.gather(*tasks)


async def run_sequential(url: str, use_system_prompt: bool = True) -> list[RequestResult]:
    """Run test cases sequentially for baseline."""
    results = []
    async with aiohttp.ClientSession() as session:
        for tc in TEST_CASES:
            result = await run_request(session, url, tc, use_system_prompt)
            results.append(result)
    return results


def print_results(results: list[RequestResult], label: str):
    """Print summary of results."""
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    ttfts = [r.time_to_first_token for r in results if not r.error]
    totals = [r.total_time for r in results if not r.error]
    tokens = [r.token_count for r in results if not r.error]
    
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    
    if ttfts:
        print(f"\n  Time to First Token (TTFT):")
        print(f"    Min:   {min(ttfts)*1000:7.0f} ms")
        print(f"    Max:   {max(ttfts)*1000:7.0f} ms")
        print(f"    Mean:  {statistics.mean(ttfts)*1000:7.0f} ms")
        
        print(f"\n  Total Completion Time:")
        print(f"    Min:   {min(totals):7.2f} s")
        print(f"    Max:   {max(totals):7.2f} s")
        print(f"    Mean:  {statistics.mean(totals):7.2f} s")
        
        wall_time = max(totals)
        total_tokens = sum(tokens)
        print(f"\n  Aggregate Throughput:")
        print(f"    Wall time:     {wall_time:.2f} s")
        print(f"    Total tokens:  {total_tokens}")
        print(f"    Throughput:    {total_tokens/wall_time:.2f} tok/s")


def print_detailed_results(results: list[RequestResult]):
    """Print per-test breakdown."""
    print(f"\n{'='*60}")
    print(f" Per-Test Breakdown")
    print(f"{'='*60}")
    
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"\n  {r.test_case.name}: {status}")
        print(f"    TTFT: {r.time_to_first_token*1000:.0f}ms, Total: {r.total_time:.2f}s, Tokens: {r.token_count}")
        print(f"    Functions: {r.functions_found}")
        print(f"    Constructs: {r.constructs_found}")
        if r.failures:
            for f in r.failures:
                print(f"    ❌ {f}")
        if r.response_text:
            snippet = r.response_text[:200].replace('\n', '\\n')
            print(f"    Code: {snippet}...")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Order Management DSL")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of concurrent")
    parser.add_argument("--no-cache", action="store_true", help="Disable system prompt (no caching)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-test results")
    args = parser.parse_args()
    
    print(f"Server: {args.url}")
    print(f"Tests: {len(TEST_CASES)}")
    print(f"Mode: {'Sequential' if args.sequential else 'Concurrent'}")
    print(f"System prompt caching: {'Disabled' if args.no_cache else 'Enabled'}")
    
    use_cache = not args.no_cache
    
    print("\nRunning tests...")
    start = time.perf_counter()
    
    if args.sequential:
        results = await run_sequential(args.url, use_cache)
    else:
        results = await run_all_concurrent(args.url, use_cache)
    
    total_time = time.perf_counter() - start
    
    label = f"{'Sequential' if args.sequential else 'Concurrent'} Results"
    if not use_cache:
        label += " (no caching)"
    print_results(results, label)
    
    if args.detailed:
        print_detailed_results(results)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Score: {100*passed//len(results)}%")
    print(f"  Total wall time: {total_time:.2f}s")
    
    if passed == len(results):
        print(f"  ✓ ALL TESTS PASSED")
    else:
        print(f"  ✗ {len(results) - passed} TESTS FAILED")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
