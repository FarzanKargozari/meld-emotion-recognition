# Experiment log

The running record of this project: what was tried, what happened, what went wrong, and what was decided.
Newest state first. Numbers come from saved files unless marked as an estimate.

## Current state (31 October 2026)

- All 12 validation runs are finished.
- Selected design: `roberta_context3`, validation macro-F1 0.491 ± 0.012.
- The final test ran once, at 21:47: **test macro-F1 0.470 ± 0.010**.
- The test split is now closed. No model may be chosen or changed using these scores.
- The repository is published, and its automated checks pass.

## Final test

Source: `reports/test/roberta_context3/`, `reports/test_ledger.json` and `reports/logs/final_test_20261031_214507.log`.
Run on 2,610 test lines with `--final`, using the choices fixed under "Selection" above. The ledger's commit field is
empty because the project was not yet a git repository.

| Model | Accuracy | Macro-precision | Macro-recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Original text baseline | 59.0% | 0.399 | 0.377 | 0.383 | 0.579 |
| **`roberta_context3`, three-seed mean (headline)** | **62.4% ± 0.7%** | | | **0.470 ± 0.010** | **0.627 ± 0.007** |
| `roberta_context3`, three-seed ensemble | 63.8% | 0.481 | 0.498 | 0.486 | 0.641 |

Ensemble 95% intervals, from resampling whole dialogues: macro-F1 0.458–0.514, weighted-F1 0.618–0.662.

- Seeds: 42 scored 0.480, 7 scored 0.461, 123 scored 0.469. Seed 123 was the best on validation (0.504) but not on
  test, so choosing a single seed by its score would have been chasing noise.
- Calibration held up on unseen data: ECE fell from 0.140 to 0.037 per seed, and was 0.028 for the ensemble. The
  temperature was 1.38 and was fitted on validation only.
- Validation-to-test change in macro-F1: 0.491 to 0.470, a drop of 0.021. Accuracy and weighted-F1 rose, which suggests
  the test split has an easier mix or more neutral lines; that explanation has not been tested.

Ensemble F1 on test, against the original baseline:

| Emotion | Test lines | Baseline | Ensemble |
|---|---:|---:|---:|
| neutral | 1,256 | 0.757 | 0.782 |
| surprise | 281 | 0.477 | 0.549 |
| fear | 50 | 0.108 | 0.259 |
| sadness | 208 | 0.251 | 0.402 |
| joy | 402 | 0.547 | 0.618 |
| disgust | 68 | 0.144 | 0.298 |
| anger | 345 | 0.400 | 0.495 |

## Setup

| Item | Value |
|---|---|
| Machine | MacBookPro17,1, 8 GB unified memory |
| Software | torch 2.10.0, transformers 5.22.1, tokenizers 0.23.3, scikit-learn 1.9.1 |
| Model | `distilroberta-base`, revision `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b` |
| Data | MELD text: 9,989 train, 1,109 validation (114 dialogues), 2,610 test |
| Labels | neutral, surprise, fear, sadness, joy, disgust, anger |
| Training | AdamW, learning rate 2e-5, 10% warmup, gradient clip 1.0, inverse-square-root class weights |
| Length | 4 rounds, patience 2, best round chosen by validation macro-F1 |
| Seeds | 42, 7, 123 |

The context designs use batch size 8 with 2 accumulation steps (effective batch 16). The single-line design uses
batch size 16. Context is past lines only, and truncation only ever removes the oldest context.

## Validation results

Source: `reports/validation/comparison.md`, `runs.json`, `gaps.json` and `history.json`.
Regenerate with `scripts/guide.sh compare`, then `.venv/bin/python scripts/record_validation.py`.

| Design | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|
| `distilroberta_context3` | 0.452 ± 0.015 | 0.596 ± 0.012 | 60.3% ± 1.1% |
| `distilroberta_context3_speakers` | 0.444 ± 0.021 | 0.588 ± 0.008 | 59.4% ± 0.8% |
| `distilroberta_utterance` | 0.433 ± 0.009 | 0.581 ± 0.001 | 59.0% ± 0.3% |

Previous best validation macro-F1, from the original models: 0.395.

### Each run

| Design | Seed | Best round | Macro-F1 | Accuracy | Device |
|---|---:|---:|---:|---:|---|
| `distilroberta_utterance` | 42 | 4 | 0.444 | 59.1% | MPS |
| `distilroberta_utterance` | 7 | 3 | 0.430 | 59.2% | CPU |
| `distilroberta_utterance` | 123 | 3 | 0.427 | 58.6% | CPU |
| `distilroberta_context3` | 42 | 4 | 0.434 | 59.1% | MPS for rounds 1–2, CPU for rounds 3–4 |
| `distilroberta_context3` | 7 | 3 | 0.460 | 61.0% | CPU |
| `distilroberta_context3` | 123 | 4 | 0.460 | 60.9% | CPU |
| `distilroberta_context3_speakers` | 42 | 4 | 0.433 | 59.4% | CPU |
| `distilroberta_context3_speakers` | 7 | 3 | 0.432 | 58.6% | CPU |
| `distilroberta_context3_speakers` | 123 | 4 | 0.468 | 60.1% | CPU |

`config.json` for `distilroberta_context3` seed 42 says CPU because a resumed run rewrites that file. This table is
the correct record of its mixed devices.

### Are the gaps real?

Paired bootstrap over whole dialogues, 2,000 resamples, comparing the three-seed means.

| Comparison | Gap | 95% interval | Share of resamples above 0 |
|---|---:|---:|---:|
| History vs single line | +0.018 | -0.001 to +0.038 | 96% |
| History + speakers vs single line | +0.011 | -0.014 to +0.035 | 79% |
| History vs history + speakers | +0.007 | -0.015 to +0.029 | 75% |

History probably helps, but the interval just includes zero. Speaker names show no measurable benefit.

### Per-class F1, mean over seeds

| Design | neutral | surprise | fear | sadness | joy | disgust | anger |
|---|---:|---:|---:|---:|---:|---:|---:|
| `distilroberta_utterance` | 0.752 | 0.612 | 0.164 | 0.360 | 0.538 | 0.213 | 0.396 |
| `distilroberta_context3` | 0.757 | 0.633 | 0.196 | 0.397 | 0.558 | 0.210 | 0.411 |
| `distilroberta_context3_speakers` | 0.749 | 0.625 | 0.189 | 0.405 | 0.540 | 0.200 | 0.404 |

### Combining seeds

Averaging the probabilities of three seeds gives macro-F1 0.430 (single line), 0.457 (history) and 0.445 (history +
speakers). All nine together give 0.448. The history ensemble gains about 0.005 for three times the model size, so a
single model is preferred.

## Upgrade test: `roberta-base`

Run on 31 October 2026. Same design as the leader (`configs/roberta_context3.json`), seed 42, CPU.

**Pass mark, written before any score was seen:** validation macro-F1 of at least 0.468.

- That is the highest single run so far, and it sits above the leader's mean plus one standard deviation (0.467).
- At or above 0.468: run seeds 7 and 123, then compare the three-seed means.
- Below 0.468: keep `distilroberta_context3` and go to the final test.

Short test before the full run: 320 lines for 1 round on CPU, peak memory 3.5 GB, no memory trouble. Its score (0.077)
means nothing at that size. The full run started at 17:05 at 0.9 seconds per step, about twice the time of
`distilroberta-base`. Output is saved in `reports/logs/`.

**Seed 42 result (finished 18:20): macro-F1 0.485, weighted-F1 0.604, accuracy 60.1%, best round 4.** Rounds scored
0.400, 0.456, 0.473 and 0.485, for 74 minutes on CPU. This passes the mark, so seeds 7 and 123 run under the rule above.
One seed is not proof: `distilroberta_context3` seeds ranged from 0.434 to 0.460.

**Three-seed result (finished 20:55, no crashes):**

| Seed | Best round | Macro-F1 | Weighted-F1 | Accuracy | Minutes |
|---:|---:|---:|---:|---:|---:|
| 42 | 4 | 0.485 | 0.604 | 60.1% | 74 |
| 7 | 4 | 0.484 | 0.608 | 60.9% | 78 |
| 123 | 4 | 0.504 | 0.609 | 61.1% | 75 |
| Mean | | **0.491 ± 0.012** | 0.607 ± 0.003 | 60.7% ± 0.5% | |

- Gap over `distilroberta_context3`: +0.039, 95% interval +0.012 to +0.067, above zero in 99.9% of resamples.
- Most of the gain is on rare emotions. F1 for fear rose from 0.196 to 0.308, disgust from 0.210 to 0.303, anger from
  0.411 to 0.446. Neutral fell slightly, from 0.757 to 0.741.
- Averaging the three calibrated seeds gives macro-F1 0.509 and accuracy 62.6% (`reports/validation/roberta_context3/`).
  That is a gain of about 0.018, against 0.005 on `distilroberta-base`, so the single-model preference no longer holds
  automatically. Without calibration the same average scores 0.511 (`gaps.json`).
- Calibration: average confidence error (ECE) fell from 0.143 to 0.041 per seed, and 0.023 for the ensemble. The
  temperature was fitted on these same validation lines, so these are optimistic until checked on the test split.
- The evaluation command was rehearsed on validation first, so the single test run should not fail partway.

**Selection: `roberta_context3`.** The headline test figure will be the three-seed mean, with the ensemble reported
beside it. Both were fixed here, before the test split was read.

These intervals treat the validation dialogues as the only source of doubt. Four designs were compared on the same
validation set, so the winner's validation score is slightly optimistic. That is why the test run still matters.

## What went wrong, and what changed

Order of events. Times are local on 31 October unless stated.

1. **Overnight, no progress in 5.5 hours.** Run 2 stalled from memory thrash on the Mac GPU (MPS). Context inputs
   average 42 tokens against 13 for single lines, and 88% of batches padded past 64 tokens. Fix: batch size 8 with 2 accumulation steps, and evaluation batch size 16 (down from 32 in
   `evaluate.py`).
2. **Slowdown to over 3 seconds per step.** Fix: empty the MPS cache every 100 steps, report speed per 100 steps,
   and free other memory, which cut swap from 7.5 GB to about 0.5 GB.
3. **09:55, stopped by SIGTERM.** The cause is unknown.
4. **11:28 and 12:33, `RuntimeError: Training loss became non-finite` on MPS.** The second happened around step
   700–800 of round 3 of run 2. The saved checkpoint was checked and was healthy. The root cause is not proven; MPS
   under memory pressure is the likely one. First fix: skip rare invalid batches, stop after more than 5. Second fix:
   move to CPU.
5. **CPU switch.** On the same 1,600-line test, CPU took 0.44 seconds per step against 1.47 on MPS, with stable swap.
   Everything after the switch ran without a crash.
6. **16:56, all 9 runs finished.**

## Disclosures for the write-up

- The original models' test results were seen once before this work began. The old test-set baseline is accuracy
  0.590, macro-F1 0.383, weighted-F1 0.579 (`reports/baselines/text_report.txt`). The test split is therefore not perfectly
  fresh, and the write-up must say so.
- Devices are mixed across runs, as listed above.
- `distilroberta_context3` seed 42 was attempted more than once. The same design and seed reached 0.412 and 0.431 at
  round 2 in different attempts, a difference of about 0.02 from the attempt alone. Only the finished attempt is kept.
- A resumed run restores the model, optimizer and schedule, but not the random state used for dropout, so a resumed
  run is not an exact replay of an uninterrupted one.
- Terminal output from these 9 runs was not saved to a file. Per-round numbers are kept in
  `reports/validation/history.json`; per-100-step speed lines are not. Training runs from now on write to
  `reports/logs/`.
- The final test used the three-seed models as selected; the score was not used to change anything afterwards.
- `distilroberta_utterance` seed 42 ran on MPS, and that design was not the one selected.

## Decisions

| Decision | Why |
|---|---|
| Select on validation macro-F1 only | Treats rare emotions fairly and protects the test split |
| Past-only context | The model must not see the future of a conversation |
| Avoid checkpoints that may have trained on MELD | That would leak test data into the model |
| Train on CPU | Faster and more stable than MPS on this 8 GB Mac |
| Prefer a single model over an ensemble (`distilroberta-base` only) | The gain was too small for three times the size |
| Select `roberta_context3` | Highest three-seed mean, and the gap over the runner-up excludes zero |
| Headline the three-seed mean, report the ensemble beside it | Fixed before reading the test split, to avoid choosing the flattering number |

## Cleanup (31 October 2026)

- The first-stage scripts, README and requirements moved to `experiments/baselines/`, and their reports to
  `reports/baselines/`. After the move, `inspect_data.py` passed and `evaluate_models.py --split validation` reproduced
  the recorded validation scores. It did not read the test split.
- All lint warnings are fixed, and all 37 tests pass.
- Added `scripts/error_analysis.py`. It describes test predictions after the model was final and changes nothing.
- Added a README, model card, GPL-3.0 licence file, Dockerfile and GitHub Actions workflow.
- MELD licence: the GitHub API reports GPL-3.0 for `declare-lab/MELD`. No MELD data or weights are in the repository.
- Clean install: a copy of only the publishable files (73 files, 544 KB) installed from scratch in a new environment,
  with no data folder. Lint was clean and 37 of 37 tests passed.
- Docker: the image built and ran as a non-root user with a mounted model. `/health` answered ok, `/predict` answered,
  and blank text was rejected with status 422.
- Scan of the publishable copy: no dashboard token, local network address or personal name, and no data, model or
  checkpoint files outside `reports/`.
- GitHub Actions: lint, 37 tests and the Docker build all passed on the first run.

## Next

1. Optional: compare with an LLM on sample lines the project creates, without using the closed test split for choices.
