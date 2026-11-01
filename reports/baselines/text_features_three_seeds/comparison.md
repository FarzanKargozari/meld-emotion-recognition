# Text features: three-seed validation comparison

| Seed | Original text macro-F1 | MiniLM macro-F1 |
|---|---:|---:|
| 42 | 0.375 | 0.383 |
| 7 | 0.371 | 0.401 |
| 123 | 0.384 | 0.382 |
| **Mean** | **0.377** | **0.389** |
| Sample standard deviation | 0.007 | 0.011 |

Values are rounded for display; aggregate statistics were calculated from the saved full-precision metrics.

MiniLM improves mean validation macro-F1 by 0.0119 (about 1.2 points on a 0–100 F1 scale), winning two of the three matched seeds. Its variation across seeds is larger. Mean accuracy is 54.7% for MiniLM versus 55.1% for the original text features; mean weighted-F1 is 0.535 versus 0.537.

This is a modest validation advantage in the selected metric, not a clear improvement on every measure. The best MiniLM run, 0.401, should not be presented as the average result. Three seeds and one validation split do not establish statistical significance or an improvement on unseen data. No new test evaluation was run.

All six runs used MPS and the same recorded optimization, weighting, hidden width, dropout, and stopping settings. The input dimensions and therefore parameter counts differ. Checkpoints were selected independently within each run using validation macro-F1.
