"""Measure DFU-LiteNet TFLite models on an Android phone with Google's official TFLite benchmark tool.

Usage:  python3 run_phone_benchmark.py <folder with .tflite files>
Needs:  adb, a phone with USB debugging on, and tools/android_aarch64_benchmark_model.
Output: phone_benchmark.csv (+ phone_benchmark_raw.txt with the tool's full output).
"""
import csv, random, re, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).parent
TOOL = HERE / "tools" / "android_aarch64_benchmark_model"
REMOTE = "/data/local/tmp/dfu"
RUNS, WARMUP, ROUNDS, COOLDOWN_S = 100, 10, 5, 3
# Each ROUND runs every (model, setting) once in a shuffled order, so phone heating and clock-speed changes over time
# affect all models alike. The reported number is the median over rounds.
# Phones mix fast "big" cores and slow "little" cores; Android may place a run on either, which changed timings ~4x in testing.
# Every run is therefore pinned with taskset to a fixed set of cores (Galaxy A35 / Exynos 1380: cpu4-7 big = f0, cpu0-3 little = 0f).
BIG, LITTLE = "f0", "0f"
CONFIGS = [  # name, core mask, extra flags
    ("big core, 1 thread (XNNPACK)", BIG, ["--num_threads=1", "--use_xnnpack=true"]),
    ("big cores, 4 threads (XNNPACK)", BIG, ["--num_threads=4", "--use_xnnpack=true"]),
    ("big core, 1 thread (built-in kernels)", BIG, ["--num_threads=1", "--use_xnnpack=false"]),
    ("little core, 1 thread (XNNPACK)", LITTLE, ["--num_threads=1", "--use_xnnpack=true"]),
    ("GPU delegate", BIG, ["--use_gpu=true"]),
]

def adb(*args, check=True):
    r = subprocess.run(["adb", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"adb {' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return r.stdout

def main(model_dir):
    models = sorted(Path(model_dir).glob("*.tflite"))
    assert models, f"no .tflite files in {model_dir}"
    devices = [l for l in adb("devices").splitlines()[1:] if l.strip().endswith("device")]
    assert len(devices) == 1, f"connect exactly one phone with USB debugging allowed (found: {devices})"
    info = {k: adb("shell", "getprop", p).strip() for k, p in
            [("phone", "ro.product.model"), ("maker", "ro.product.manufacturer"), ("chip", "ro.board.platform"),
             ("soc", "ro.soc.model"), ("android", "ro.build.version.release"), ("abi", "ro.product.cpu.abi")]}
    print("Phone:", info)
    assert info["abi"].startswith("arm64"), "the benchmark tool needs a 64-bit ARM phone"

    adb("shell", "mkdir", "-p", REMOTE)
    adb("push", str(TOOL), f"{REMOTE}/benchmark")
    adb("shell", "chmod", "+x", f"{REMOTE}/benchmark")
    rows, raw = [], []
    for m in models:
        adb("push", str(m), f"{REMOTE}/{m.name}")
    tests = [(m, c) for m in models for c in CONFIGS]
    res = {(m.name, c[0]): {"med": [], "p95": [], "mem": [], "err": ""} for m, c in tests}
    rng = random.Random(42)
    for rnd in range(ROUNDS):
        order = tests[:]; rng.shuffle(order)
        for m, (cname, cores, flags) in order:
            r = res[(m.name, cname)]
            if r["err"]:
                continue
            cmd = ["taskset", cores, f"{REMOTE}/benchmark", f"--graph={REMOTE}/{m.name}", f"--num_runs={RUNS}",
                   f"--warmup_runs={WARMUP}", "--report_peak_memory_footprint=true", *flags]
            time.sleep(COOLDOWN_S)
            out = subprocess.run(["adb", "shell", " ".join(cmd)], capture_output=True, text=True, timeout=600)
            text = out.stdout + out.stderr
            raw.append(f"===== round {rnd + 1} | {m.name} | {cname}\n{text}")
            stats = re.findall(r"count=\d+ .*?median=([\d.]+) p95=([\d.]+)", text)
            if not stats:
                r["err"] = next((l for l in text.splitlines() if re.search(r"error|fail", l, re.I)), "failed")
                continue
            r["med"].append(float(stats[-1][0]) / 1000); r["p95"].append(float(stats[-1][1]) / 1000)
            mem = re.search(r"overall=([\d.]+)", text)
            if mem: r["mem"].append(float(mem.group(1)))
        print(f"round {rnd + 1}/{ROUNDS} done", flush=True)
    for m, (cname, cores, flags) in tests:
        r = res[(m.name, cname)]; ok = not r["err"]
        rows.append({**info, "model_file": m.name, "size_mb": round(m.stat().st_size / 1e6, 2), "config": cname,
                     "median_ms": round(statistics.median(r["med"]), 2) if ok else "",
                     "min_ms": round(min(r["med"]), 2) if ok else "", "max_ms": round(max(r["med"]), 2) if ok else "",
                     "p95_ms": round(statistics.median(r["p95"]), 2) if ok else "",
                     "peak_memory_mb": round(max(r["mem"]), 1) if r["mem"] else "",
                     "status": "ok" if ok else f"failed: {r['err'].strip()[:150]}"})
        print(f"{m.name:32s} {cname:38s} " + (f"{rows[-1]['median_ms']:7.2f} ms  (range over rounds {rows[-1]['min_ms']}-{rows[-1]['max_ms']})"
                                              if ok else "FAILED: " + r["err"].strip()[:70]), flush=True)
    adb("shell", "rm", "-r", REMOTE, check=False)

    with open(HERE / "phone_benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (HERE / "phone_benchmark_raw.txt").write_text("\n".join(raw))
    print("\nSaved phone_benchmark.csv and phone_benchmark_raw.txt")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "models")
