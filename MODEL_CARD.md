# Model card: `roberta_context3`

## Summary

| Item | Value |
|---|---|
| Task | Predict one of 7 emotions for a line of dialogue: neutral, surprise, fear, sadness, joy, disgust, anger |
| Base model | `roberta-base`, fine-tuned with a classification head |
| Input | The target line and up to 3 earlier lines from the same conversation |
| Output | Calibrated probabilities for the 7 emotions |
| Language | English |
| Weights | Not distributed; train them with `configs/roberta_context3.json` |

## Intended use

This is a research project for studying emotion recognition in dialogue and honest model evaluation.
It may suit offline analysis of scripted or fictional dialogue, where mistakes carry no consequences for anyone.

## Out of scope

Do not use it to make or support decisions about people. That includes hiring, education, health or mental-health
assessment, insurance, policing, workplace monitoring and customer scoring. A predicted emotion is a guess from text,
and it is not evidence of how a person feels.

## Training data

MELD text from the TV show *Friends*: 9,989 training lines, 1,109 validation lines and 2,610 test lines, using the
official splits. About 42% of validation lines and 48% of test lines are neutral. Fear and disgust are rare.

## Evaluation

All figures are on the test split. Details are in [`reports/test/roberta_context3/`](reports/test/roberta_context3).

| Measure | Three-seed mean | Three-seed ensemble |
|---|---:|---:|
| Macro-F1 | 0.470 ± 0.010 | 0.486 (95% interval 0.458–0.514) |
| Weighted-F1 | 0.627 ± 0.007 | 0.641 (95% interval 0.618–0.662) |
| Accuracy | 62.4% ± 0.7% | 63.8% |
| Calibration error (ECE) | 0.037 | 0.028 |

Ensemble F1 by emotion:

| neutral | surprise | fear | sadness | joy | disgust | anger |
|---:|---:|---:|---:|---:|---:|---:|
| 0.782 | 0.549 | 0.259 | 0.402 | 0.618 | 0.298 | 0.495 |

## Limits and risks

- **Weak on rare emotions:** fear and disgust F1 are below 0.30, and their scores rest on only 50 and 68 test lines.
- **Narrow data:** one scripted American sitcom, from a small group of main characters. Performance on other speakers,
  dialects, cultures or real conversations has not been measured.
- **No fairness testing:** results have not been broken down by speaker group.
- **Text only:** sarcasm and tone that depend on voice or expression are often missed.
- **Calibration may not transfer:** the temperature was fitted on MELD validation data.
- **Test split reuse:** the project's first models were scored on this split once before this work began.
