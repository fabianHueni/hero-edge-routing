# HERO Notebooks
These notebooks support dataset preparation and the analysis of HERO experiment results.

## Setup
This project uses `uv` as the package manager. Ensure `uv` is installed, then navigate to this folder and run:

```bash
uv sync
```

## Notebooks
The following notebooks are available in this folder:
- **`00_preprocess_datasets.ipynb`**: Preprocess the raw public available dataset to match our experiments (add id column and create balanced subdataset)
- **`01_generate_lorem_ipsum_dataset.ipynb`**: Generate our introduced lorem ipsum dataset
- **`02_analyse_baseline_experiments.ipynb`**: Analyse the baseline experiments to understand the performance of on device and cloud inference
- **`03_analyse_queueing_behaviour.ipynb`**: Analyse the queueing behaviour of the conduced experiments
- **`04_design_routing_policies.ipynb`**: Design proposed routing policies
- **`05_analyse_HERO_performance.ipynb`**: Analyse the performance of HERO