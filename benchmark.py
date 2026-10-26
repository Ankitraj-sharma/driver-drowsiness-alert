"""
Performance Benchmarking Suite for DrowsyGuard AI.
Measures latency and FPS throughput for EAR, MAR, and solvePnP algorithms.
"""

import time
import numpy as np
from metrics import calculate_ear, calculate_mar, estimate_head_pose


def run_benchmark(iterations: int = 1000):
    print(f"[*] Running {iterations} iterations of kinematic solvers...")

    # Sample eye landmarks
    eye = [(0.0, 10.0), (3.0, 14.0), (7.0, 14.0), (10.0, 10.0), (7.0, 6.0), (3.0, 6.0)]
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = calculate_ear(eye)
    ear_time = (time.perf_counter() - t0) * 1000 / iterations

    # Sample mouth landmarks
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = calculate_mar((5.0, 15.0), (5.0, 5.0), (1.0, 10.0), (9.0, 10.0))
    mar_time = (time.perf_counter() - t0) * 1000 / iterations

    # Head pose points
    pnp_2d = [(320, 240), (320, 350), (270, 200), (370, 200), (290, 300), (350, 300)]
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = estimate_head_pose(pnp_2d, (480, 640))
    pose_time = (time.perf_counter() - t0) * 1000 / iterations

    print(f"[-] EAR Average Latency:       {ear_time:.4f} ms")
    print(f"[-] MAR Average Latency:       {mar_time:.4f} ms")
    print(f"[-] solvePnP Average Latency:  {pose_time:.4f} ms")
    total_time = ear_time + mar_time + pose_time
    print(f"[+] Total Kinematic Compute:   {total_time:.4f} ms ({1000/total_time:.0f} FPS theoretical)")


if __name__ == "__main__":
    run_benchmark()
