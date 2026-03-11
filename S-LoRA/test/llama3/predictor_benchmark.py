"""
predictor_benchmark.py
----------------------
Send a diverse set of requests to a running slora server, then flush the prediction
CSV via /exit_finetuning.

Usage:
    python predictor_benchmark.py --port 9000 --out llama3_predictions.csv
    python predictor_benchmark.py --port 9001 --out mixtral_predictions.csv --base_url http://localhost

The script sends 60 requests in waves of 10, with varied prompt lengths and content
types (code, math, prose, lists) to exercise different MoE routing patterns.
After all requests complete, it POSTs /exit_finetuning to flush the tracker CSV.
"""

import argparse
import asyncio
import json
import time
import aiohttp


# ---------------------------------------------------------------------------
# Prompt templates — varied content stresses different expert routing patterns
# ---------------------------------------------------------------------------

CODE_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list.",
    "Implement a linked list class in Python with insert, delete, and search methods.",
    "Write a recursive Fibonacci function with memoization using functools.lru_cache.",
    "Implement quicksort in Python. Explain each step with inline comments.",
    "Write a decorator in Python that measures and prints execution time of functions.",
    "Implement a simple stack using a Python list with push, pop, and peek methods.",
    "Write a Python class for a min-heap with insert and extract_min operations.",
    "Implement breadth-first search on an adjacency list representation of a graph.",
    "Write a Python generator that yields all prime numbers up to a given limit.",
    "Implement matrix multiplication in Python without using NumPy.",
    "Write a Python context manager that times code blocks and logs the duration.",
    "Implement a simple tokenizer in Python that splits text on whitespace and punctuation.",
]

MATH_PROMPTS = [
    "Solve this system of linear equations: 2x + 3y = 12 and 4x - y = 5. Show steps.",
    "Find the derivative of f(x) = x^3 * sin(x) using the product rule. Show work.",
    "Compute the integral of x^2 * e^x dx using integration by parts twice.",
    "Prove by induction that the sum of first n natural numbers is n*(n+1)/2.",
    "Find all eigenvalues of the matrix [[3, 1], [0, 2]]. Show characteristic polynomial.",
    "Calculate the probability that at least one of three fair dice shows a 6.",
    "Find the Taylor series expansion of cos(x) around x=0 up to the x^6 term.",
    "Solve the recurrence relation T(n) = 2T(n/2) + n using the master theorem.",
    "Compute the determinant of a 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]] by cofactor expansion.",
    "Find the limit of (sin x)/x as x approaches 0 using L'Hopital's rule.",
    "Compute the eigendecomposition of the symmetric matrix [[4, 2], [2, 3]].",
    "Show that the square root of 2 is irrational using proof by contradiction.",
]

PROSE_PROMPTS = [
    "Explain the difference between supervised and unsupervised machine learning in simple terms.",
    "Describe how the immune system recognizes and destroys pathogens. Be detailed.",
    "Explain why the sky appears blue during the day and red at sunset. Use physics.",
    "Describe the process of how stars form from clouds of gas and dust in space.",
    "Explain the difference between strong AI and weak AI, and why the distinction matters.",
    "Describe the economic concept of comparative advantage and give a concrete example.",
    "Explain how HTTPS encrypts web traffic and why certificate authorities matter.",
    "Describe the major differences between RNA and DNA in structure and function.",
    "Explain how garbage collection works in a managed language like Python or Java.",
    "Describe the key phases of the software development lifecycle in a modern agile team.",
    "Explain the concept of entropy in thermodynamics and information theory.",
    "Describe how neural networks learn through backpropagation and gradient descent.",
]

LIST_PROMPTS = [
    "List the top 10 sorting algorithms with their average and worst-case time complexities.",
    "List 8 key differences between SQL and NoSQL databases with examples of each.",
    "List the 7 OSI network model layers and describe what happens at each layer.",
    "List 10 important design patterns in object-oriented programming with brief descriptions.",
    "List the main programming paradigms (functional, OOP, procedural, etc.) with key traits.",
    "List 8 common security vulnerabilities in web applications (OWASP top 10 style).",
    "List the key differences between TCP and UDP, and when to use each one.",
    "List 10 important Linux commands every developer should know with examples.",
    "List the main types of machine learning algorithms grouped by learning strategy.",
    "List 8 differences between interpreted and compiled programming languages with examples.",
    "List the major HTTP status code categories (2xx, 3xx, 4xx, 5xx) with common examples.",
    "List 10 git commands and their uses for managing branches and resolving conflicts.",
]

# ---------------------------------------------------------------------------
# Build request list with varied token counts
# ---------------------------------------------------------------------------

def build_requests():
    """Return list of (prompt, max_new_tokens) targeting ~50/100/200/400/800 token prompts."""
    all_prompts = []
    # Short prompts (~50 tokens) — repeat a short question
    for p in CODE_PROMPTS[:3]:
        all_prompts.append((p[:80], 50))   # truncate to keep short
    for p in MATH_PROMPTS[:3]:
        all_prompts.append((p[:80], 50))

    # Medium prompts (~100-150 tokens)
    for p in PROSE_PROMPTS[:4]:
        all_prompts.append((p, 50))
    for p in LIST_PROMPTS[:4]:
        all_prompts.append((p, 50))

    # Longer prompts (~200 tokens) — concatenate two prompts
    for i in range(0, 8, 2):
        combined = CODE_PROMPTS[i] + " Also, " + MATH_PROMPTS[i]
        all_prompts.append((combined, 50))

    # Longer prompts (~400 tokens)
    for i in range(0, 6, 2):
        combined = (PROSE_PROMPTS[i] + " Furthermore, " + PROSE_PROMPTS[i+1]
                    + " Finally, " + LIST_PROMPTS[i])
        all_prompts.append((combined, 50))

    # Longest prompts (~600-800 tokens)
    for i in range(0, 4):
        combined = (CODE_PROMPTS[i] + "\n\n" + MATH_PROMPTS[i] + "\n\n"
                    + PROSE_PROMPTS[i] + "\n\n" + LIST_PROMPTS[i])
        all_prompts.append((combined, 50))

    return all_prompts


# ---------------------------------------------------------------------------
# Async HTTP helpers
# ---------------------------------------------------------------------------

async def send_request(session, base_url, prompt, max_new_tokens, request_id):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
    }
    t0 = time.time()
    try:
        async with session.post(f"{base_url}/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=300)) as resp:
            result = await resp.json()
            latency = time.time() - t0
            status = "ok" if resp.status == 200 else f"err_{resp.status}"
            return request_id, status, latency
    except Exception as e:
        return request_id, f"exception:{e}", time.time() - t0


async def run_benchmark(base_url, prompts, wave_size=10):
    results = []
    connector = aiohttp.TCPConnector(limit=wave_size * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Send in waves of wave_size
        for wave_start in range(0, len(prompts), wave_size):
            wave = prompts[wave_start:wave_start + wave_size]
            tasks = [
                send_request(session, base_url, p, max_tok, wave_start + i)
                for i, (p, max_tok) in enumerate(wave)
            ]
            wave_results = await asyncio.gather(*tasks)
            results.extend(wave_results)
            ok = sum(1 for _, s, _ in wave_results if s == "ok")
            avg_lat = sum(l for _, _, l in wave_results) / len(wave_results)
            print(f"  Wave {wave_start//wave_size + 1}: {ok}/{len(wave)} ok, "
                  f"avg latency {avg_lat:.2f}s")

        # Flush prediction CSV via /exit_finetuning
        print("Flushing prediction CSV via /exit_finetuning ...")
        try:
            async with session.post(f"{base_url}/exit_finetuning",
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                print(f"  /exit_finetuning → {resp.status}")
        except Exception as e:
            print(f"  /exit_finetuning failed: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--wave_size", type=int, default=10)
    parser.add_argument("--out", default="predictions.csv",
                        help="Output path hint (CSV is written by the server, "
                             "this arg is just for reference)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    prompts = build_requests()
    print(f"Running {len(prompts)} requests in waves of {args.wave_size} → {base_url}")
    print(f"Server will write prediction CSV on /exit_finetuning flush.")
    print(f"Reference output file: {args.out}\n")

    t_start = time.time()
    results = asyncio.run(run_benchmark(base_url, prompts, wave_size=args.wave_size))
    elapsed = time.time() - t_start

    ok = sum(1 for _, s, _ in results if s == "ok")
    print(f"\nDone: {ok}/{len(results)} ok in {elapsed:.1f}s")
    print(f"The server's prediction CSV will be written to the server's working directory.")
    print(f"Look for: prediction_stats_*.csv")


if __name__ == "__main__":
    main()
