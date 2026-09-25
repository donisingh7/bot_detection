# Social Bot Detection — ML Research Experiments

Machine-learning experiments for detecting **bot-controlled social-media accounts** from tabular account/profile features.

This repository preserves multiple modelling approaches and their actual notebook outputs, including approaches that did **not** perform strongly. The purpose is to document the research process honestly rather than present a production-ready detector.

## Repository contents

| File | Purpose |
| --- | --- |
| `Approach_1_Random_Forest_Model.ipynb` | Baseline preprocessing + Random Forest classifier |
| `Approach_2.ipynb` | XGBoost, LightGBM, and stacking experiments |
| `bot_detection_data.csv` | Dataset used by the notebooks |
| `Presentation_Detecting_Bot_Controlled_Accounts_on_Social_Media.pdf` | Project presentation |
| `Presentation_Detecting_Bot_Controlled_Accounts_on_Social_Media.pptx` | Editable presentation source |

## Modelling flow

```text
Social-account tabular data
        │
        ▼
Preprocessing
missing values / categorical encoding
        │
        ▼
Train/test split
        │
        ├── Random Forest baseline
        ├── XGBoost
        ├── LightGBM
        └── Stacking classifier
             └── Logistic Regression meta-model
        │
        ▼
Classification report + confusion matrix
```

## Results captured in the notebooks

The stored experiments show that the available feature set did **not** separate bots and non-bots strongly.

- Random Forest baseline: approximately **50% test accuracy** in the stored run.
- Additional XGBoost/LightGBM/stacking experiments also remained around chance-level performance in several stored outputs.

These results are retained because they are useful research evidence: they show that changing classifiers alone is insufficient when the underlying feature signal is weak or the dataset/labels do not provide enough separation.

## What this project demonstrates

- tabular ML preprocessing
- scikit-learn pipelines
- Random Forest classification
- XGBoost and LightGBM experimentation
- stacking ensembles
- classification reports and confusion matrices
- evaluation of weak-signal datasets without hiding negative results

## Reproduce

```bash
python -m venv .venv
pip install -r requirements.txt
jupyter notebook
```

Open either notebook from the repository root so `bot_detection_data.csv` resolves correctly.

## Research limitations

- The stored dataset/features appear to have limited predictive signal for the target label.
- Accuracy near 50% means this should **not** be treated as a deployable bot detector.
- A stronger follow-up would improve feature engineering, validate label quality, check leakage/duplicates, add temporal/graph/network features, and evaluate on an external holdout dataset.
- The notebooks are preserved as research artifacts, not production code.

## Author

**Doni Singh Agrawal**

Academic ML research project exploring social-media bot detection and ensemble-learning approaches.
