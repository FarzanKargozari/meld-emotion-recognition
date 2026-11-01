# MELD emotion classification on an Apple M1

> **Archived.** This is the project's first stage: small classifiers on published features. It is kept for the record and is not the current model. See the [main README](../../README.md). Run the commands below from the repository root.

A small, locally trained comparison of text, audio, and combined text/audio classifiers for seven-class emotion recognition on MELD. The project uses published feature vectors and trains its own neural classifiers.

**Result:** text-only performed best: **0.383 test macro-F1** and **59.0% test accuracy**. Combining text and audio by concatenation did not improve performance in this experiment.

## What the project demonstrates

- Matching text, audio, and labels by split, dialogue ID, and utterance ID.
- Checking feature dimensions, missing IDs, duplicate IDs, and finite values.
- Fitting normalization and class weights using training data only.
- Training three modality baselines with the same training procedure.
- Selecting checkpoints and the preferred model by validation macro-F1.
- Evaluating fixed models on the held-out test split and reporting class imbalance.
- Running locally on an Apple M1 MacBook Pro with 8 GB unified memory and PyTorch MPS.

## Data

MELD contains conversations from *Friends* with seven emotion labels: neutral, surprise, fear, sadness, joy, disgust, and anger.

| Split | Examples | Purpose |
|---|---:|---|
| Train | 9,989 | Learn parameters, scaling, and class weights |
| Validation | 1,109 | Choose checkpoints and preferred modality |
| Test | 2,610 | Final evaluation of fixed experiments |

The official feature archive supplies 300-dimensional averaged GloVe text features and 1,611-dimensional selected audio features for emotion classification. The combined input has 1,911 dimensions. No visual features are used.

Data sources: [MELD repository](https://github.com/declare-lab/MELD) and [official feature download](https://huggingface.co/datasets/declare-lab/MELD/tree/main).

## Method

Each classifier uses `input → 256 → 7` layers, with ReLU and dropout of 0.3. Only the input width changes: 300 for text, 1,611 for audio, and 1,911 for both. This means model parameter counts differ between modalities.

- AdamW; learning rate 0.001; weight decay 0.01.
- Batch size 32; float32; seed 42.
- Weighted cross-entropy: inverse square root of training class frequency, normalized to mean weight 1.
- StandardScaler fitted on training features only and reused at evaluation.
- Up to 30 epochs; early stopping after five epochs without improved validation macro-F1.
- Best saved checkpoint chosen by validation macro-F1, not the last epoch.

The classifiers treat utterances independently; they do not model dialogue context or speaker identity. Training happens on MPS when available, with CPU support as well.

## Results

### Validation

| Model | Best epoch | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|---:|
| Majority baseline | — | 0.085 | 0.252 | 42.4% |
| Text | 5 | **0.375** | **0.542** | **56.3%** |
| Audio | 2 | 0.210 | 0.331 | 32.2% |
| Text + audio | 2 | 0.308 | 0.451 | 42.3% |

Text-only was selected before test evaluation. The majority baseline always predicts the most frequent training emotion, neutral.

### Test

| Model | Macro-F1 | Weighted-F1 | Accuracy |
|---|---:|---:|---:|
| Majority baseline | 0.093 | 0.313 | 48.1% |
| Text | **0.383** | **0.579** | **59.0%** |
| Audio | 0.196 | 0.344 | 33.0% |
| Text + audio | 0.293 | 0.463 | 42.8% |

Text-only improves test accuracy over the majority baseline by **10.9 percentage points**. Its macro-F1 improvement indicates better performance across emotions than always predicting neutral. However, the text model remains weak on fear (F1 0.108) and disgust (F1 0.144), compared with neutral (F1 0.757).

The combined model is better than audio alone, but worse than text alone. These results support a narrow conclusion: this feature-concatenation baseline did not benefit from adding audio. They do not establish that audio is generally unhelpful.

Aggregate metrics and per-class results are in `reports/baselines/`. Local checkpoints, scalers, predictions, and run logs are in `results/` and excluded from version control.

## Reproduce

Use Python 3.11. The original experiments used Python 3.11.14, PyTorch 2.10.0, and the Apple M1 GPU. MPS availability depends on the execution environment. The scripts also work on CPU; exact results may differ across devices and software versions.

### 1. Environment

Run these commands from this project folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/baselines/requirements.txt
```

`requirements.txt` pins direct dependencies. `environment-lock.txt` records all installed packages in the original environment, including optional packages unused by these scripts.

### 2. Features

Download `MELD.Features.Models.tar.gz` from the official source linked above. If it is in Downloads, extract only these three files:

```bash
mkdir -p data
tar -xzf "$HOME/Downloads/MELD.Features.Models.tar.gz" -C data --strip-components=2 MELD.Features.Models/features/data_emotion.p MELD.Features.Models/features/text_glove_average_emotion.pkl MELD.Features.Models/features/audio_embeddings_feature_selection_emotion.pkl
```

The scripts read these files from `data/`. They are legacy Python pickle files; use the official source rather than arbitrary pickle downloads. Data is not distributed with this project.

### 3. Inspect and train

```bash
python experiments/baselines/inspect_data.py
python experiments/baselines/train_model.py --modality text
python experiments/baselines/train_model.py --modality audio
python experiments/baselines/train_model.py --modality both
```

Each run prints its saved results folder. To explicitly select CPU, add `--device cpu` to a training command. Training can be verified with `--epochs 1`, but that is not the full experiment.

### 4. Evaluate saved checkpoints

For the original local experiment, the defaults point to the three recorded run folders:

```bash
python experiments/baselines/evaluate_models.py
```

For new training runs, substitute the actual folder names printed by training:

```bash
python experiments/baselines/evaluate_models.py --text-run TEXT_RUN_FOLDER --audio-run AUDIO_RUN_FOLDER --both-run BOTH_RUN_FOLDER
```

The evaluator selects the preferred model from saved validation scores before evaluating test data. `--split validation` instead verifies that reloaded models reproduce their recorded validation metrics. Do not tune model settings based on the test results.

## Follow-up: late fusion on validation

After the initial test report, a bounded exploratory experiment combined the saved text and audio models' softmax probabilities. The four audio weights were specified before running the comparison: 0%, 5%, 10%, and 20%. No checkpoints were retrained, no scalers were refitted, and test data was not evaluated in this follow-up.

| Text contribution | Audio contribution | Validation macro-F1 | Weighted-F1 | Accuracy |
|---|---|---:|---:|---:|
| 100% | 0% | **0.375** | 0.542 | 56.3% |
| 95% | 5% | 0.367 | 0.540 | 56.3% |
| 90% | 10% | 0.370 | 0.542 | 56.6% |
| 80% | 20% | 0.366 | 0.542 | 56.7% |

Text-only remained best by validation macro-F1. A small accuracy increase at higher audio weights did not translate into better macro-F1. The probabilities were not calibrated, so this experiment assesses this simple weighted-average method, not every possible fusion strategy. The original models were trained on MPS; this comparison was run on CPU and first verified reproduction of both models' saved validation metrics.

Run the same experiment with:

```bash
python experiments/baselines/late_fusion.py
```

It automatically uses MPS when available, otherwise CPU; add `--device cpu` to match the recorded inference device. For different checkpoints, use `--text-run` and `--audio-run` with the respective saved run folders. Selection uses validation macro-F1, with ties favoring less audio. Aggregate follow-up results are in `reports/baselines/late_fusion_validation/`.

## Follow-up: frozen MiniLM text features

This experiment replaces the 300-dimensional averaged GloVe features with 384-dimensional sentence embeddings from `sentence-transformers/all-MiniLM-L6-v2`. It uses the same supplied sentence text and official train/validation splits, without adding dialogue context. The encoder is frozen; only the emotion classifier is trained.

```bash
python -m pip install -r experiments/baselines/requirements-minilm.txt
python experiments/baselines/extract_minilm.py
python experiments/baselines/train_model.py --modality text --text-features minilm
```

Extraction saves features and model provenance under `data/minilm_features/`; it refuses to overwrite an existing feature folder. If extraction has already finished, run only the training command. The loader verifies the source-file hash and matches IDs and labels before training. Test sentences are not encoded by this workflow.

The classifier uses the same hidden width, dropout, optimization settings, class weighting, seed, scaling procedure, and early stopping rule as the original text model. Its input width changes from 300 to 384, so its parameter count also changes. New runs are stored separately under `results/minilm_text_*` and record the encoder revision in their configuration. Compare validation macro-F1 with the original text baseline, **0.375**. No improvement is assumed in advance.

The original `evaluate_models.py` and `late_fusion.py` workflows above are for the GloVe-based checkpoints; they have not yet been extended to MiniLM checkpoints.

The first MiniLM run (seed 42, best epoch 5) produced validation macro-F1 **0.383**, weighted-F1 **0.527**, and accuracy **53.9%**. Compared with the original text run, macro-F1 rose from 0.375, but weighted-F1 and accuracy fell from 0.542 and 56.3%. This is a small validation tradeoff from one run, not evidence of a reliable overall gain. MiniLM has not been evaluated on test data.

To check sensitivity to random initialization, repeat both representations with two additional seeds, keeping all other settings fixed:

```bash
python experiments/baselines/train_model.py --modality text --seed 7
python experiments/baselines/train_model.py --modality text --text-features minilm --seed 7
python experiments/baselines/train_model.py --modality text --seed 123
python experiments/baselines/train_model.py --modality text --text-features minilm --seed 123
```

Together with the original seed-42 runs, this will give three runs per representation. Compare mean validation macro-F1 and variation across all three seeds, rather than choosing the best seed. These repeat runs are now complete. Full-precision metrics and the aggregate comparison are in `reports/baselines/text_features_three_seeds/`.

Across seeds 42, 7, and 123, original text achieved mean validation macro-F1 **0.377 ± 0.007**, while MiniLM achieved **0.389 ± 0.011** (mean ± sample standard deviation across seeds). MiniLM won two of three matched seeds, but mean accuracy was slightly lower: 54.7% versus 55.1%. Mean weighted-F1 was 0.535 versus 0.537. This is a modest validation advantage on macro-F1, without a significance claim or a new test evaluation.

## Planned follow-up: stronger dropout for MiniLM

The next controlled comparison changes hidden-layer dropout from 0.3 to 0.5 for the MiniLM classifier. Feature files, hidden width, optimizer, class weighting, batch size, splits, and early stopping remain unchanged. Repeat the same three seeds:

```bash
python experiments/baselines/train_model.py --modality text --text-features minilm --dropout 0.5 --seed 42
python experiments/baselines/train_model.py --modality text --text-features minilm --dropout 0.5 --seed 7
python experiments/baselines/train_model.py --modality text --text-features minilm --dropout 0.5 --seed 123
```

The script prints and records the requested dropout probability. Each run saves to a new folder. Omitting `--dropout` keeps the original 0.3 setting. Compare the three-run mean validation macro-F1 with **0.389** for MiniLM at dropout 0.3, and report seed variation alongside it. These dropout-0.5 runs have not yet been performed; no improvement is assumed.

## Limits and future work

- The original multimodal comparison used one seed. The later text-feature comparison uses three seeds, with sample standard deviation but no confidence intervals or statistical significance test.
- The project trains classifiers on published features; it does not train text/audio encoders end to end or process raw recordings.
- The upstream feature-generation and feature-selection procedures have not been independently audited. Claims about train-only preprocessing apply to this project's scaling and weighting steps.
- Rare emotions remain difficult; accuracy alone can hide that weakness.
- The simple fusion architecture and shared hyperparameters may not suit both modalities equally.
- Performance on scripted TV dialogue does not establish performance on real-world conversations.

Future experiments could compare modality-specific projections, late fusion, contextual text features, and multiple seeds, with decisions made on validation data. This test evaluation is already known; further exploration must be reported transparently rather than described as a fresh untouched test.

## Reference

Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., and Mihalcea, R. (2019). *MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations*. ACL 2019. See the official repository for dataset documentation and terms.
