# Selective Decoder Routing for Surface Code Data

This research project studies **when to use correlated matching instead of PyMatching** on real hardware-derived surface-code syndrome data. A router sees allowed syndrome and experiment features, chooses one of two existing decoder predictions for each shot, and aims for **low logical error rate (LER) with less unnecessary correlated-matching use**. The router is a selector, not a quantum error-correction decoder.

Logistic Regression, Random Forest, XGBoost, and a variational quantum classifier (VQC) are candidate routers for the **same routing objective**. This is not primarily an ML-versus-QML benchmark. The main system figure will be **LER versus correlated-matching usage** as the routing threshold changes. No model has been trained and no project results are claimed yet.

The [Hugging Face dataset](https://huggingface.co/datasets/Bauxitiego/surface-code-syndromes) reformats [Google Quantum AI's original experimental data](https://zenodo.org/records/6804040). It includes train, validation, and test splits and stored per-shot decoder predictions. The full dataset is not kept in this repository.

```text
Real syndrome data
       ↓
Feature extraction
       ↓
Routing model
       ↓
Choose PyMatching or correlated matching
       ↓
Final decoder prediction
       ↓
System evaluation
```

## Day 1 and evaluation logic

Start by checking the dataset schema and split definitions on a small development sample. For each shot, compare each decoder prediction with the true logical label. The four outcome groups are:

| PyMatching | Correlated matching | Meaning |
| --- | --- | --- |
| Correct | Correct | Both succeed |
| Wrong | Correct | **Rescue opportunity**; positive routing target |
| Correct | Wrong | Switching would harm this shot |
| Wrong | Wrong | Neither decoder succeeds |

The rescue opportunity is `PM wrong AND CM correct`. The **oracle LER** is the fraction `PM wrong AND CM wrong`: even a perfect selector of these two predictions fails there. PyMatching LER and correlated-matching LER are each the fraction of shots their respective decoder gets wrong. Use decoder predictions and the true label only for targets and evaluation, never as router input features.

## Planned notebooks (you create these)

There are **no `.ipynb` files in the repository yet**. Create these six notebooks in `notebooks/` when ready. Each should have a title, purpose, short Colab setup section, and placeholders that call reusable code from `src/router/`:

| Notebook | Responsibility |
| --- | --- |
| `00_dataset_and_feasibility.ipynb` | Day 1: inspect schema, rows, and splits; calculate four outcome groups, both decoder LERs, rescue opportunity, and oracle LER. |
| `01_feature_engineering.ipynb` | Validate detection-event count and density, round activity, active rounds, maximum round activity, distance, rounds, and basis. Mask padded syndrome positions with `round_widths`. |
| `02_classical_routers.ipynb` | Try a Dummy baseline, Logistic Regression, Random Forest, and XGBoost on the same routing target. |
| `03_qml_router.ipynb` | Explore a VQC/QML routing candidate using compact features for the same routing objective. |
| `04_routing_tradeoff.ipynb` | Sweep thresholds on validation data and plot LER against correlated-matching usage. |
| `05_final_evaluation.ipynb` | Evaluate the chosen router once on untouched test samples against always-PyMatching, always-correlated-matching, and oracle routing. |

For a Colab runtime, the intended setup is:

```bash
git clone https://github.com/rifahnanjiba02-arch/selective-decoder-routing.git
cd selective-decoder-routing
pip install -e .
```

Install optional QML dependencies only when working on the VQC notebook: `pip install -e '.[qml]'`. Keep notebook code focused on exploration and presentation; add reusable logic to `src/router/`.

## Repository layout

```text
selective-decoder-routing/
├── README.md
├── requirements.txt            # convenience: installs pyproject dependencies
├── pyproject.toml
├── Makefile
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── models.yaml
├── notebooks/                 # planned, no notebooks committed yet
│   ├── 00_dataset_and_feasibility.ipynb  (planned)
│   ├── 01_feature_engineering.ipynb      (planned)
│   ├── 02_classical_routers.ipynb        (planned)
│   ├── 03_qml_router.ipynb               (planned)
│   ├── 04_routing_tradeoff.ipynb         (planned)
│   └── 05_final_evaluation.ipynb         (planned)
├── src/router/
│   ├── __init__.py
│   ├── data.py                # dataset and development sampling
│   ├── decoders.py            # decoder outcome accounting
│   ├── labels.py              # positive routing target
│   ├── features.py            # leakage-safe feature planning
│   ├── routing.py             # future threshold and selection logic
│   ├── evaluate.py            # future system metrics
│   ├── plots.py               # future shared plots
│   └── models/
│       ├── __init__.py
│       ├── classical.py
│       └── qml.py
├── results/                   # planned: feasibility/, figures/, tables/, model_scores/
├── artifacts/                 # local only: features/, models/; Git-ignored
├── app/streamlit_app.py       # reserved; dashboard not implemented
├── docs/
│   ├── PROJECT_GUIDE.md
│   ├── DATA_DICTIONARY.md
│   └── EXPERIMENT_PROTOCOL.md
├── paper/                     # planned writing and figures
└── tests/                     # tests added as research code is implemented
```

The Python package, configuration templates, and protocol are the starting point for a 15-day implementation schedule. Begin with feasibility, then feature engineering and candidate routers, then validation threshold selection, and finally one test-set evaluation. See [the project guide](docs/PROJECT_GUIDE.md), [data dictionary](docs/DATA_DICTIONARY.md), and [experiment protocol](docs/EXPERIMENT_PROTOCOL.md).
