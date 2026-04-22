# AutoResearch Agent Instructions — Manga Adaptation Classifier

## Objective

Maximize **validation ROC-AUC** on the manga → anime adaptation classification task.
Secondary metric: F1 score (reported but not used for keep/discard decisions).

## Task Description

Binary classification: given manga features known **before** adaptation,
predict whether the manga will be adapted into an anime.

Target: `adapted_to_anime` (1 = adapted, 0 = not adapted)

## Rules

1. You may **ONLY** modify `model.py`
2. `prepare.py` and `run.py` are **FROZEN** — do not touch them
3. `build_model()` must return an sklearn-compatible classifier with `predict_proba()`
4. Training + evaluation must complete in **under 90 seconds** on CPU
5. No additional data sources or external downloads
6. **Do NOT evaluate on the test set** during the loop — only use val_auc for decisions

## Workflow

```
1. Read current model.py
2. Propose one modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check val_auc in output
6. If improved:  keep change, note new best
7. If worse:     revert model.py to previous version
8. Repeat from step 1 — aim for at least 8 iterations
```

## Available Features (adaptation-independent only)

Numeric:
- num_volumes, num_chapters      — size of the work
- start_year                     — era of publication
- run_years                      — how long it ran
- in_jump_magazine (0/1)         — published in Jump family
- in_major_magazine (0/1)        — published in major magazine

Genre flags (18 binary columns):
- genre_action, genre_adventure, genre_comedy, genre_drama, genre_fantasy,
  genre_horror, genre_mystery, genre_romance, genre_scifi, genre_slice_of_life,
  genre_sports, genre_supernatural, genre_thriller, genre_shounen, genre_shoujo,
  genre_seinen, genre_josei, genre_ecchi

Categorical (one-hot encoded):
- media_type   — manga / manhwa / novel / one_shot / doujinshi
- status       — finished / currently_publishing
- nsfw         — white / gray / black

## Ideas to Explore

* Classifiers: LogisticRegression, GradientBoostingClassifier, HistGradientBoostingClassifier, SVC, XGBClassifier
* Imbalance handling: class_weight="balanced", SMOTE oversampling, threshold tuning
* Feature engineering: interaction between in_jump_magazine × genre_shounen, decade bins from start_year
* Preprocessing: RobustScaler, QuantileTransformer, feature selection via SelectKBest
* Ensembles: VotingClassifier, StackingClassifier

## What NOT to Do

* Do not modify `prepare.py` or `run.py`
* Do not add features that are consequences of adaptation (score, rank, popularity, member counts)
* Do not evaluate on the test set (X_test, y_test) — it is locked until final reporting
* Do not hard-code validation data into the model
* Do not change the function signature of `build_model()`
