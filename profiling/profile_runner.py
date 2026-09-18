import cProfile
import pstats
import time
import tracemalloc
from unoptimized_version.engine import FeatureEngineUnoptimized


def generate_mock_data():
    """Generates deterministic mock datasets for repeatable benchmarks."""
    import random
    random.seed(42)

    # 200 samples with 50 dimensions each for matrix computation
    data_matrix = [[random.uniform(0.0, 100.0) for _ in range(50)] for _ in range(200)]
    # 5,000 numerical features
    feature_list = [random.uniform(-50.0, 150.0) for _ in range(5000)]
    # 10,000 ID records with repeated elements
    id_list = [random.randint(1000, 3000) for _ in range(10000)]

    return data_matrix, feature_list, id_list


def run_baseline_benchmark():
    engine = FeatureEngineUnoptimized()
    data_matrix, feature_list, id_list = generate_mock_data()

    print("=" * 60)
    print("RUNNING BASELINE (UNOPTIMIZED) PERFORMANCE PROFILE")
    print("=" * 60)

    tracemalloc.start()
    start_time = time.perf_counter()

    # Execute all 3 tasks
    _ = engine.compute_pairwise_distances(data_matrix)
    _ = engine.normalize_and_filter(feature_list, threshold=2.0)
    _ = engine.find_unique_ids(id_list)

    end_time = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time = end_time - start_time
    print(f"\n[BASELINE METRICS]")
    print(f"Total Execution Time: {total_time:.4f} seconds")
    print(f"Peak Memory Usage:    {peak_mem / 1024 / 1024:.2f} MB\n")

    print("--- Detailed cProfile Function Breakdown ---")
    profiler = cProfile.Profile()
    profiler.enable()

    _ = engine.compute_pairwise_distances(data_matrix)
    _ = engine.normalize_and_filter(feature_list, threshold=2.0)
    _ = engine.find_unique_ids(id_list)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)


if __name__ == "__main__":
    run_baseline_benchmark()