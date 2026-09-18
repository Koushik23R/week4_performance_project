# AI/ML Feature Engineering Performance Engine

A high-performance feature engineering and computational module designed to identify, profile, and optimize performance bottlenecks in Python-based data processing workflows.

---

## 1. Project Overview

In Machine Learning pipelines, processing high-dimensional feature matrices and numerical datasets can easily suffer from execution slowdowns due to unvectorized loops, redundant calculation of loop invariants, and inefficient data structure lookups.

This application implements three core computational tasks:
1. **Pairwise Distance Matrix:** Computing Euclidean distance matrices across multi-dimensional feature vectors.
2. **Feature Normalization & Filtering:** Calculating Z-score normalization and filtering outliers.
3. **Record Deduplication:** Extracting unique identifier lists from raw sequence logs.

The project demonstrates a full performance engineering workflow:
- Establishing baseline metrics using `cProfile`, `pstats`, and `tracemalloc`.
- Identifying structural bottlenecks in execution time and memory usage.
- Refactoring algorithms to reduce computational complexity.
- Verifying functional parity using automated unit testing (`unittest`).

---

## 2. Key Optimization Results

| Metric | Baseline (Unoptimized) | Optimized Implementation | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Time** | 36.8606 seconds | 1.4536 seconds | **25.36x Faster** |
| **Peak Memory** | 1.27 MB | 0.81 MB | **36.2% Reduction** |
| **Function Calls** | 25,162,188 calls | 47,210 calls | **99.8% Reduction** |
| **Functional Parity** | Baseline reference | 100% Identical Output | **PASS (3/3 Tests)** |

---

## 3. Optimization Summary

1. **Loop Invariant Pre-Computation:** In feature normalization, `mean` and `std` were recalculated inside the iteration loop across 5,000 items ($O(N^2)$ complexity). Pre-computing these statistics once outside the loop reduced function calls from ~25M to under 50K.
2. **Symmetric Distance Calculation:** In pairwise distance calculation, computing $D(i, j)$ and $D(j, i)$ redundantly was replaced by processing upper-triangle elements only ($D(i, j) = D(j, i)$), cutting distance math operations by 50%.
3. **Set-Based Hash Lookups:** In sequence deduplication, checking existing elements via linear search on Python lists ($O(N)$ lookup) was replaced with an auxiliary `set()` lookup ($O(1)$ constant time complexity).

---

## 4. Repository Structure

```text
week4_performance_project/
├── unoptimized_version/
│   ├── __init__.py
│   └── engine.py           # Baseline engine containing computational bottlenecks
├── optimized_version/
│   ├── __init__.py
│   └── engine.py           # Optimized engine with algorithmic improvements
├── profiling/
│   └── profile_runner.py    # cProfile, timeit, and tracemalloc profiling suite
├── tests/
│   ├── __init__.py
│   └── test_parity.py      # Automated parity test suite verifying identical output
├── docs/
│   └── PERFORMANCE_REPORT.md
├── README.md
├── requirements.txt
└── .gitignore
```
## 5. How to Run Tests and Profiling

### Prerequisites
- Python 3.8+ (Uses standard library packages: `cProfile`, `pstats`, `tracemalloc`, `timeit`, `unittest`).

### Step 1: Run Functional Parity Tests
To verify that the optimized code yields identical results to the unoptimized baseline:

```bash
python -m unittest discover -s tests -v
```

### Step 2: Run Performance Benchmark & Profiling Comparison
To execute the benchmark comparison and view execution times, peak memory usage, and speedup metrics:

```bash
python -m profiling.profile_runner