import os
import sys
import json
import multiprocessing as mp
from typing import Dict, List
import signal
import contextlib
import faulthandler
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def write_jsonl(filename: str, data: List[Dict]):
    """Write list of objects to jsonl file."""
    with open(filename, "w") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")


def unsafe_execute(code: str, timeout: float = 3.0):
    """Execute code in a separate process with timeout."""

    def run_code():
        try:
            exec(code, {})
            return True, None
        except Exception as e:
            return False, str(e)

    with mp.Pool(1) as pool:
        try:
            success, error = pool.apply_async(run_code).get(timeout=timeout)
            return success, error
        except mp.TimeoutError:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)


def check_test_case(sample: Dict, test_case: str) -> bool:
    """Check a single test case."""
    code = sample["completion"] + "\n" + test_case
    success, error = unsafe_execute(code)
    return success


def evaluate_sample(sample: Dict, test_cases: List[str]) -> bool:
    """Evaluate a single sample against all test cases."""
    for test_case in test_cases:
        if not check_test_case(sample, test_case):
            return False
    return True


def evaluate_functional_correctness(samples_file: str) -> Dict:
    """Evaluate functional correctness of completions."""
    # Read samples
    with open(samples_file, "r") as f:
        samples = [json.loads(line) for line in f]

    print("Reading test cases...")
    test_cases = {}
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_cases")
    for sample in samples:
        task_id = sample["task_id"]
        if task_id not in test_cases:
            test_file = os.path.join(test_dir, f"{task_id}.py")
            if os.path.exists(test_file):
                with open(test_file, "r") as f:
                    test_cases[task_id] = f.read().strip().split("\n\n")
            else:
                test_cases[task_id] = []

    print("Evaluating samples...")
    correct = 0
    total = 0

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for sample in samples:
            task_id = sample["task_id"]
            if task_id in test_cases:
                futures.append(
                    executor.submit(evaluate_sample, sample, test_cases[task_id])
                )
                total += 1

        # Collect results
        for future in as_completed(futures):
            if future.result():
                correct += 1

    # Calculate pass@k metrics
    results = {
        "pass@1": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
    }

    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <samples_file>")
        sys.exit(1)

    results = evaluate_functional_correctness(sys.argv[1])
    print(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
