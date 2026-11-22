# Corrections to 05c outputs (made locally, 2026-11-21)

- ConvNeXtTiny GFLOPs: the Kaggle profiler reported 0.03 (wrong). Recomputed locally with the same
  profiler method on the same architecture (224x224, same head): **8.757 GFLOPs**.
  Check: the same script gives MobileNetV3Small 0.116 and ResNet50 7.751, exactly matching Kaggle.
- ConvNeXtTiny CPU latency (673 ms) came from a CPU-copy fallback path, so it is not comparable with the
  other models and is left blank in the corrected tables.
- Files: efficiency_corrected.csv, summary_table_corrected.csv. Accuracy numbers are unchanged.
