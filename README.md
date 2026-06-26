# HEA-KANs

Custom utilities and worked examples for fitting **Vickers Hardness (HV)** of
High Entropy Alloys (HEAs) using
[pyKAN](https://github.com/KindXiaoming/pykan) — Kolmogorov–Arnold Networks
with learnable spline activation functions.

Dataset: Berry/Gorsse 2018 — 244 HEA compositions, 10 elemental fraction
inputs, HV output (range 100–905 HV).

---

## Library — `HEA_pyKAN.py`

| Function / Class | Description |
|---|---|
| `to_np(x)` | Detach a tensor and return a numpy array |
| `train_val_test_split(...)` | Three-way stratified split (train / val / test) with optional quantile-bin stratification |
| `best_loss_KAN` | `KAN` subclass that checkpoints the state with the best validation loss during training, restoring it at the end (early-stopping without early termination) |
| `mini_batch_train_best_model(...)` | Mini-batch Adam retraining with the same val-loss checkpointing, for use after edge pruning |
| `k_fold_val(...)` | 5-fold cross-validation using `best_loss_KAN`; each fold is split 50:50 into test/val and two models are trained (A/B swap), giving 10 evaluations per call |
| `get_best_result(study)` | Extract a summary dict from an Optuna study's best trial |
| `sort_edge_scores(model)` | Rank all edges by attribution score, lowest first |
| `remove_ranked_edges(model, n_remove)` | Remove the `n_remove` lowest-scoring active edges |
| `ablate_edges(...)` | Run a multi-phase pruning schedule, retraining after each removal and tracking (edge count, MAE) at every checkpoint. Supports `target_edges` to stop at a chosen sparsity level |
| `DEFAULT_ABLATION_SCHEDULE` | Empirically tuned three-phase schedule: coarse (−20 × 8), intermediate (−10 × 3), fine (−5 × 3) |

### Regularisation convention

pyKAN applies regularisation as:

```
loss_reg = lamb × (lamb_l1 × L1 + lamb_entropy × entropy + …)
```

`lamb` is a master multiplier, so `lamb_entropy` passed to the model is
*relative* to `lamb`.  The best-hyperparameter JSON stores the **absolute**
entropy weight (`lamb_entropy_actual = lamb × lamb_entropy_model`).  Convert
before use:

```python
lamb_entropy = params["lamb_entropy"] / params["lamb"]
```

---

## Best hyperparameters — `best_hyperparams_by_hidden_nodes.json`

Optuna search results (100 trials per hidden-node count, fixed shuffle
seed = 42) for hidden-node counts 1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30.
Each entry stores `k`, `g`, `lamb`, `lamb_entropy` (absolute), and
`model_seed`.

---

## Example notebooks

| Notebook | Description |
|---|---|
| `HEA-pyKAN-demo.ipynb` | Load best hyperparameters, run a single cross-validation, repeat across 10 shuffle seeds, and plot MAE vs seed |
| `HEA-pyKAN-optuna.ipynb` | Run an Optuna hyperparameter search for a chosen hidden-node count using the cross-validation objective; save results to JSON |
| `HEA-pyKAN-ablate.ipynb` | Edge ablation study: train a 20-hidden-node KAN, iteratively prune it across 10 dataset seeds, plot MAE vs edge count with error bars, then visualise how the network architecture evolves through a single continuous pruning run |

---

## Acknowledgements

This work builds on **pyKAN** (the `pykan` package) by Liu et al.  If you use
this repository in your research, please also cite the original KAN papers:

```
@article{liu2024kan,
  title   = {{KAN}: Kolmogorov-Arnold Networks},
  author  = {Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian
             and Halverson, James and Solja\v{c}i\'{c}, Marin and Hou, Thomas Y.
             and Tegmark, Max},
  journal = {arXiv preprint arXiv:2404.19756},
  year    = {2024}
}

@article{liu2024kan2,
  title   = {{KAN} 2.0: Kolmogorov-Arnold Networks Meet Science},
  author  = {Liu, Ziming and Zhu, Boxuan and Bright, Sean and Vaidya, Sachin
             and Guo, Haoxuan and Gao, Tao and Tegmark, Max},
  journal = {arXiv preprint arXiv:2408.10205},
  year    = {2024}
}
```

The HEA dataset is from:

```
@article{gorsse2018database,
  title   = {Database on mechanical properties of high entropy alloys and
             complex concentrated alloys},
  author  = {Gorsse, St{\'e}phane and Miracle, Daniel B. and Senkov, Oleg N.},
  journal = {Data in Brief},
  volume  = {21},
  pages   = {2664--2678},
  year    = {2018}
}
```

---

## Requirements

```
pykan
torch
numpy
pandas
scikit-learn
optuna
matplotlib
tqdm
```

Install into your environment:

```bash
pip install pykan torch numpy pandas scikit-learn optuna matplotlib tqdm
```

---

## Data

`Gorsse2018ML_Data_RawHV.csv` is included in this repository.  It is the
Berry/Gorsse 2018 HEA dataset (244 compositions, 10 elemental fraction inputs,
raw Vickers Hardness output).

Update the `DATA_PATH` variable at the top of each notebook to point at the
location where you have cloned the repo:

```python
DATA_PATH = r"C:\path\to\HEA-KANS"
DATA_FILE = "Gorsse2018ML_Data_RawHV.csv"
```
