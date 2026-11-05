# Selective Decoder Routing for Surface Code Data

This research project studies **when to use correlated matching instead of PyMatching** on real hardware-derived surface-code syndrome data. A router sees allowed syndrome and experiment features, chooses one of two existing decoder predictions for each shot, and aims for **low logical error rate (LER) with less unnecessary correlated-matching use**. The router is a selector, not a quantum error-correction decoder.

Logistic Regression, Random Forest, XGBoost, and a variational quantum classifier (VQC) are candidate routers for the **same routing objective**. This is not primarily an ML-versus-QML benchmark. The main system figure will be **LER versus correlated-matching usage** as the routing threshold changes. The current results are decoder-feasibility measurements on a training sample; no router has been trained or evaluated yet.

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

## Notebooks

The first notebook, `00_dataset_and_feasibility.ipynb`, is complete. Its saved Colab run covers the pinned 50,000-row train sample, schema checks, four decoder outcomes, five rates, and the rescue-only routing target. All 16 code cells ran in order with no error outputs. The [data dictionary](docs/DATA_DICTIONARY.md) records the schema checks, and the [combined Week 1 findings](results/feasibility/week1_findings.md) record settings, results, and limitations. Notebooks 01 and 02 contain Week 2 work outlines, while later notebooks remain planned. Each notebook should call reusable code from the `router` package inside `src/` (for example, `from router.decoders import summarize_outcomes`):

| Notebook | Responsibility |
| --- | --- |
| `00_dataset_and_feasibility.ipynb` | Week 1, complete: inspect the train sample and schema; calculate decoder outcomes, rates, and the rescue-only routing target. |
| [`01_feature_engineering.ipynb`](notebooks/01_feature_engineering.ipynb) (scaffold) | Validate detection-event count and density, round activity, active rounds, maximum round activity, distance, rounds, and basis. Mask padded syndrome positions with `round_widths`. |
| [`02_classical_routers.ipynb`](notebooks/02_classical_routers.ipynb) (scaffold) | Try a Dummy baseline, Logistic Regression, Random Forest, and XGBoost on the same routing target. |
| `03_qml_router.ipynb` | Explore a VQC/QML routing candidate using compact features for the same routing objective. |
| `04_routing_tradeoff.ipynb` | Sweep thresholds on validation data and plot LER against correlated-matching usage. |
| `05_final_evaluation.ipynb` | Evaluate the chosen router once on untouched test samples against always-PyMatching, always-correlated-matching, and oracle routing. |

For a Colab runtime, the intended setup is:

```bash
git clone https://github.com/rifahnanjiba02-arch/selective-decoder-routing.git
cd selective-decoder-routing
pip install -e .
```

The current Week 1 notebook instead downloads the required `router` files and installs its small runtime dependency set. Both setups call the same package functions.

Week 2 covers feature engineering and the first classical routers in notebooks 01 and 02. Their scaffolds, test plans, and empty results-note templates are ready; no Week 2 results have been recorded. The [Week 2 task guide](docs/WEEK_2_TASK_GUIDE.md) divides that work into four tasks, and the [Week 2 schedule](docs/WEEK_2_SCHEDULE.md) gives seven-day checkpoints.

Install optional QML dependencies only when working on the VQC notebook: `pip install -e '.[qml]'`. Keep notebook code focused on exploration and presentation; add reusable logic to the `router` package inside `src/`.

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
├── notebooks/
│   ├── 00_dataset_and_feasibility.ipynb  (complete)
│   ├── 01_feature_engineering.ipynb      (scaffold)
│   ├── 02_classical_routers.ipynb        (scaffold)
│   ├── 03_qml_router.ipynb               (planned)
│   ├── 04_routing_tradeoff.ipynb         (planned)
│   └── 05_final_evaluation.ipynb         (planned)
├── src/
│   └── router/
│       ├── __init__.py
│       ├── data.py            # dataset and development sampling
│       ├── decoders.py        # decoder outcome accounting
│       ├── labels.py          # positive routing target
│       ├── features.py        # Week 2 extractor scaffold
│       ├── routing.py         # threshold and decoder selection
│       ├── evaluate.py        # Week 2 validation-metric scaffold
│       ├── plots.py           # future shared plots
│       └── models/
│           ├── __init__.py
│           ├── classical.py
│           └── qml.py
├── results/
│   ├── README.md
│   ├── feasibility/
│   ├── figures/
│   ├── tables/
│   └── model_scores/
├── artifacts/                 # local only; Git-ignored
│   ├── features/             (planned)
│   └── models/               (planned)
├── app/
│   └── streamlit_app.py      # reserved; dashboard not implemented
├── docs/
│   ├── PROJECT_GUIDE.md
│   ├── DATA_DICTIONARY.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── WEEK_1_TASK_GUIDE.md
│   └── WEEK_1_SCHEDULE.md
├── paper/
│   └── README.md
└── tests/
    ├── test_data.py
    ├── test_features.py      (scaffold; no tests yet)
    ├── test_classical.py     (scaffold; no tests yet)
    └── test_routing.py
```

Entries marked `(planned)` are listed for the intended workflow and have not been created. Local artifacts are ignored by Git.

The Python package, configuration templates, and protocol are the starting point for the project. Week 1's feasibility work is documented in its [schedule](docs/WEEK_1_SCHEDULE.md) and [task guide](docs/WEEK_1_TASK_GUIDE.md). Week 2 plans feature engineering and classical routers; later work covers validation threshold selection and one final test-set evaluation. See also the [project guide](docs/PROJECT_GUIDE.md), [data dictionary](docs/DATA_DICTIONARY.md), and [experiment protocol](docs/EXPERIMENT_PROTOCOL.md).
