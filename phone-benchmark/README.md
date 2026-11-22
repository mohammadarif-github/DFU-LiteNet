# Phone benchmark: Samsung Galaxy A35 (SM-A356E, Exynos 1380, Android 16)

Measured on 2026-11-16 with Google's official TFLite benchmark tool (`android_aarch64_benchmark_model`), via `run_phone_benchmark.py`.

**Method**
- Each test = 100 timed runs after 10 warm-up runs.
- 5 rounds; every round tests all models and settings in a shuffled order, so phone heating affects all models alike.
- Reported value = median over the rounds.
- Every run is pinned with `taskset` to the fast cores (cpu4-7) or the slow cores (cpu0-3). Without pinning, timings varied up to about 4x.

**Model files** are the fold-0 models from notebook 09's local test. Latency depends on the graph, not the trained weights.
