import cProfile
import pstats
import time
import tracemalloc
from unoptimized_version.engine import FeatureEngineUnoptimized
from optimized_version.engine import FeatureEngineOptimized


def generate_mock_data():
    import random
    random.seed(42)
    data_matrix = [[random.uniform(0.0, 100.0) for _ in range(50)] for _ in range(200)]
    feature_list = [random.uniform(-50.0, 150.0) for _ in range(5000)]
    id_list = [random.randint(1000, 3000) for _ in range(10000)]
    return data_matrix, feature_list, id_list


def run_comparative_benchmark():
    data_matrix, feature_list, id_list = generate_mock_data()

    unopt_engine = FeatureEngineUnoptimized()
    opt_engine = FeatureEngineOptimized()

    print("=" * 65)
    print("RUNNING COMPARATIVE PERFORMANCE BENCHMARK")
    print("=" * 65)

    # 1. Baseline Run
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = unopt_engine.compute_pairwise_distances(data_matrix)
    _ = unopt_engine.normalize_and_filter(feature_list, threshold=2.0)
    _ = unopt_engine.find_unique_ids(id_list)
    t1 = time.perf_counter()
    _, unopt_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    unopt_time = t1 - t0

    # 2. Optimized Run
    tracemalloc.start()
    t0 = time.perf_counter()
    _ = opt_engine.compute_pairwise_distances(data_matrix)
    _ = opt_engine.normalize_and_filter(feature_list, threshold=2.0)
    _ = opt_engine.find_unique_ids(id_list)
    t1 = time.perf_counter()
    _, opt_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    opt_time = t1 - t0

    speedup = unopt_time / opt_time if opt_time > 0 else 0

    print(f"\n[BENCHMARK RESULTS]")
    print(f"Unoptimized Execution Time: {unopt_time:.4f} s | Peak Memory: {unopt_mem / 1024 / 1024:.2f} MB")
    print(f"Optimized Execution Time:   {opt_time:.4f} s | Peak Memory: {opt_mem / 1024 / 1024:.2f} MB")
    print(f"Overall Speedup Factor:     {speedup:.2f}x Faster\n")


if __name__ == "__main__":
    run_comparative_benchmark()