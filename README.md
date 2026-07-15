---
title: On Device Vs Cloud Llm Inference
emoji: 📉
colorFrom: indigo
colorTo: yellow
sdk: static
pinned: false
---

# HERO: Hybrid Edge-Cloud Routing Orchestrator for Browser-Based LLM Inference

This repository contains the research artefact accompanying a conference paper on **HERO** — a hybrid request-routing
system that dynamically dispatches language model inference jobs to either an on-device (browser-local) model or a
cloud-hosted model, with the goal of minimising end-to-end latency under realistic load conditions.

---

## Overview

Recent advances in WebGPU and the [Transformers.js](https://github.com/xenova/transformers.js) library have made it
feasible to run small language models (SLMs) entirely in the browser. While on-device inference avoids network
round-trips, cloud inference can be faster when the local device is saturated. HERO bridges both worlds by continuously
estimating the expected latency on each backend and routing each incoming request to whichever backend is predicted to
respond sooner.

The project covers the full research pipeline:

1. **Experiment Framework** — a browser application (`index.html` + `src/`) that generates synthetic request streams with
   Poisson-distributed inter-arrival times, routes requests to on-device or cloud backends, measures latency, and
   records structured results.
2. **Dataset preparation & analysis** — Jupyter notebooks (`notebooks/`) that pre-process datasets, characterise
   baseline performance, model queueing behaviour, and evaluate HERO's routing policies.
3. **Results** — raw and aggregated CSV artefacts from experiments conducted on multiple real devices.

### Routing Strategies

The `RequestManager` supports the following routing strategies:

| Strategy        | Description                                                                                                                            |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `always_cloud`  | All requests are sent to the cloud backend                                                                                             |
| `always_device` | All requests are processed on-device                                                                                                   |
| `probabilistic` | Each request goes to the cloud with a configurable probability                                                                         |
| `roundrobin`    | Requests alternate between cloud and device                                                                                            |
| `hero`          | Our **HERO** policy — predicts latency on both backends using a Bayesian linear model (Thompson Sampling) and routes to the faster one |

`hero` is the core of HERO: it maintains online linear performance models for both backends and uses Thompson Sampling
to balance exploration and exploitation as load conditions change.

---

## Repository Structure

```
.
├── index.html                  # Browser-based experiment runner (entry point)
├── src/                        # JavaScript source code
│   ├── main.js                 # Application logic: wires UI, scheduler, and request manager
│   ├── scheduler.js            # JobScheduler: emits requests with configurable arrival patterns
│   ├── requestManager.js       # RequestManager: routing logic, queue management, statistics
│   ├── evaluator.js            # Evaluator: exact-match accuracy and throughput metrics
│   ├── datasetLoader.js        # Loads datasets from CSV files in the browser
│   ├── utils.js                # Shared utilities (timing helpers, sleep, etc.)
│   └── services/
│       ├── cloudService.js     # Cloud inference via OpenRouter API
│       └── onDeviceService.js  # On-device inference via Transformers.js + WebGPU
├── data/                       # Datasets used in experiments (CSV)
│   ├── boolq_validation.csv    # BoolQ yes/no question answering
│   ├── ag_news_test.csv        # AG News topic classification
│   ├── imdb_dataset.csv        # IMDB sentiment classification
│   ├── spam_ham_dataset.csv    # Spam/ham email classification
│   └── lorem_ipsum_dataset.csv # Synthetic Lorem Ipsum dataset (length-controlled prompts)
├── notebooks/                  # Python analysis environment
│   ├── 00_preprocess_datasets.ipynb        # Dataset pre-processing and balancing
│   ├── 01_generate_lorem_ipsum_dataset.ipynb # Generate the synthetic Lorem Ipsum dataset
│   ├── 02_analyse_baseline_experiments.ipynb # Baseline on-device vs cloud latency analysis
│   ├── 03_analyse_queueing_behaviour.ipynb   # Queueing dynamics under varying arrival rates
│   ├── 04_design_routing_policies.ipynb      # Policy design and parameter calibration
│   ├── 05_analyse_HERO_performance.ipynb     # End-to-end HERO performance evaluation
│   ├── utils.py                              # Shared Python utilities
│   ├── pyproject.toml                        # Python project metadata and dependencies
│   └── plots/                               # Generated figures (output of notebooks)
└── results/                    # Experiment result artefacts (CSV)
    ├── baseline_experiments/   # Per-device baseline measurements (cloud, desktop, m2air, m4pro, zenbook, …)
    ├── policy_experiments/     # Results from routing policy experiments
    └── policy_simulations/     # Simulation results comparing policies across datasets
```

---

## Setup

### Browser Experiment Runner

The experiment runner is a static web application. No build step is required. A short tutorial video is available on [YouTube](https://www.youtube.com/watch?v=zvwa7xPAYkc&t=64s)

**Prerequisites**

- A modern browser with **WebGPU** support (we recommend brave or chrome, which were used for our experiments).
- An [OpenRouter](https://openrouter.ai/) API key for cloud inference.

**Steps**

1. Serve the repository root over HTTP (browsers block local file imports for ES modules):
   ```bash
   # Python ≥ 3
   python -m http.server 8080
   # or Node.js
   npx serve .
   ```
2. Open `http://localhost:8080` in your browser.
3. In the UI, enter your **OpenRouter API key** and select the desired cloud model.
4. Optionally configure the on-device model name.
5. Set the arrival rate (λ), dataset, routing strategy, and if desired, HERO linear-model parameters.
6. Click **Load Model** to download and cache the on-device model (first run may take several minutes depending on model
   size and network speed).
7. Click **Start Experiment** to begin. Results are logged to the browser console and can be exported as CSV.

### Python Analysis Notebooks

The notebooks use [`uv`](https://github.com/astral-sh/uv) for dependency management and require **Python 3.11**.

**Prerequisites**

- Python 3.11
- `uv` ([installation guide](https://github.com/astral-sh/uv#installation))

**Steps**

```bash
# Navigate to the notebooks directory
cd notebooks

# Install dependencies (creates a virtual environment automatically)
uv sync

# Launch Jupyter (or use your preferred IDE with implemented jupyter support)
uv run jupyter notebook
```

All notebooks are self-contained and read from the `../results/` and `../data/` directories relative to the `notebooks/`
folder. Run them in numerical order (`00_` → `05_`) when reproducing the full analysis from scratch.

---

## Datasets

| Dataset             | Task                                | Source                                                                      |
|---------------------|-------------------------------------|-----------------------------------------------------------------------------|
| BoolQ               | Yes/no question answering           | [Google / HuggingFace](https://huggingface.co/datasets/google/boolq)        |
| AG News             | 4-class news topic classification   | [HuggingFace](https://huggingface.co/datasets/ag_news)                      |
| IMDB (not in paper) | Binary sentiment classification     | [Kaggle / HuggingFace](https://huggingface.co/datasets/stanfordnlp/imdb)    |
| Spam/Ham            | Binary spam detection               | [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) |
| Lorem Ipsum         | Synthetic length-controlled prompts | Generated (see `notebooks/01_generate_lorem_ipsum_dataset.ipynb`)           |

Raw datasets were pre-processed using `notebooks/00_preprocess_datasets.ipynb`, which adds a stable `id` column and
creates balanced sub-samples.

---

## Results Structure

Each experiment produces two CSV files per configuration:

- `*_raw_experiment_*.csv` — one row per request, including job ID, routing decision, queueing time, inference time,
  total latency, model response, and exact-match accuracy.
- `*_stats_experiment_*.csv` — aggregated summary statistics for the experiment run.

File names encode the dataset, arrival rate (λ), on-device model, cloud model, routing strategy, arrival pattern, and
timestamp, e.g.:

```
boolq_lambda10_raw_experiment_Llama-3-2-1B-Instruct-ONNX_meta-llama-llama-3-2-1b-instruct_jseq_exponential-arrival_2026-02-26T20-21-56.csv
```

Baseline results are organised by device under `results/baseline_experiments/` (
subdirectories: `cloud`, `desktop`, `m2air`, `m4pro`, `zenbook`, etc.).

---

## Key Dependencies

| Component           | Technology                                                            |
|---------------------|-----------------------------------------------------------------------|
| On-device inference | [Transformers.js](https://github.com/xenova/transformers.js) (WebGPU) |
| Cloud inference     | [OpenRouter](https://openrouter.ai/) REST API                         |
| UI styling          | [Tailwind CSS](https://tailwindcss.com/) (CDN)                        |
| Data analysis       | Python 3.11, pandas, NumPy, SciPy, Matplotlib, Seaborn, Jupyter       |
| Package management  | [uv](https://github.com/astral-sh/uv)                                 |

---
## Note

Please note that during development the HERP policy was preliminary called jseq. This is why in some files (csv and notebooks) 
variables and columns can still be called jseq. We keep this to not break the analysis of existing results.
---


## Citation

> *Citation details to be added upon paper publication.*