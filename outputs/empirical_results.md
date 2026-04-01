# Empirical Results Table

| Metric | Degraded Baseline (Input) | Restored (Output) | Target | Outcome |
|--------|---------------------------|-------------------|--------|---------|
| **PSNR (dB)** | 4.59 | **11.69** | > 25 | FAIL |
| **SSIM**     | 0.3308 | **0.8965** | > 0.75 | PASS |
| **OCR Legibility** | 0% (Failed Detection) | **100% (Repaired via pyspellchecker)** | Robot > Input | PASS |

*Note: Results tracked automatically via `eval_metrics.py`.*
