## Author: Koushik R
## Role: Junior Python Developer (YuvaIntern)

# AI/ML Feature Engineering Performance Engine

A Python project designed to demonstrate how performance bottlenecks in data-processing pipelines can be identified, measured, and optimized while preserving the same functional behavior.

## Overview

This repository compares an unoptimized feature-engineering implementation with an optimized version to show the practical impact of algorithmic improvements in real ML-style workloads.

The project evaluates three core operations:

- Pairwise Euclidean distance calculation for feature vectors
- Z-score normalization and filtering of numerical data
- Deduplication of ID lists while preserving order

The goal is to highlight how redundant computation, repeated loop invariant work, and inefficient list lookups can significantly affect runtime and memory usage.

## Why This Matters

In machine learning and data engineering pipelines, small inefficiencies become large costs at scale. When processing large datasets, even modest algorithmic improvements can produce major gains in performance and scalability.

This project demonstrates a full performance engineering workflow:

- profiling execution time and memory usage
- identifying bottlenecks in algorithm design
- refactoring for lower complexity
- validating output parity with automated tests

## Repository Structure

```text
week4_performance_project/
├── optimized_version/
│   └── engine.py             # Optimized implementation
├── profiling/
│   └── profile_runner.py     # Benchmark and profiling runner
├── tests/
│   └── test_parity.py        # Functional parity verification
├── unoptimized_version/
│   └── engine.py             # Baseline implementation with bottlenecks
├── .gitignore
├── README.md
└── .idea/
```

## Performance Highlights

The following figures reflect a representative benchmark run on the project’s sample dataset:

| Metric | Unoptimized | Optimized | Improvement |
| --- | ---: | ---: | ---: |
| Execution Time | 36.8606 s | 1.4536 s | 25.36x faster |
| Peak Memory | 1.27 MB | 0.81 MB | 36.2% reduction |
| Function Calls | 25,162,188 | 47,210 | 99.8% reduction |
| Functional Parity | Baseline reference | 100% identical output | Pass |

## Optimization Strategies Applied

1. Loop invariant pre-computation
   - Mean and standard deviation were calculated once instead of repeatedly inside loops.

2. Reduced redundant distance work
   - Pairwise distances were computed only once per symmetric pair and mirrored across the matrix.

3. Efficient duplicate handling
   - Linear list membership checks were replaced with a hash-based `set` for constant-time membership testing.

## How to Run

### Prerequisites

- Python 3.8 or newer
- Standard library only: `cProfile`, `pstats`, `tracemalloc`, `time`, and `unittest`

### 1. Run the unit tests

```bash
python -m unittest discover -s tests -v
```

### 2. Run the benchmark comparison

```bash
python -m profiling.profile_runner
```

This prints execution time, memory usage, and the observed speed improvement between the unoptimized and optimized implementations.

## Expected Result

The optimized version should produce the same outputs as the baseline implementation while reducing runtime and memory consumption substantially.

## Project Goal

This project is a hands-on example of performance engineering in Python, demonstrating that meaningful optimization is not only about writing faster code but also about understanding complexity, measuring bottlenecks, and validating correctness.

---

Developed as a performance-analysis project for Python-based feature engineering workloads.