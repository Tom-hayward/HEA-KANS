"""
HEA-pyKAN.py
============
Custom utilities for fitting HEA Vickers Hardness data with pyKAN.

Regularisation convention
--------------------------
pyKAN's KAN.fit() uses a master weight `lamb` that multiplies all penalty terms:

    reg = lamb * (lamb_l1 * L1 + lamb_entropy * entropy + ...)

`lamb_entropy` passed to fit() is therefore *relative* to `lamb`.  The
Optuna search in this project optimises an *absolute* entropy weight
(`lamb_entropy_actual = lamb * lamb_entropy_model`), which is what is
stored in the best-hyperparameter JSON files.

When loading from a JSON file, convert before passing to the model:

    lamb            = params["lamb"]
    lamb_entropy    = params["lamb_entropy"] / lamb   # stored value is absolute
    model.best_loss_fit(..., lamb=lamb, lamb_entropy=lamb_entropy)
"""

import copy
import os

import numpy as np
import sklearn.model_selection
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from kan import KAN, LBFGS
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_np(x):
    """Detach a tensor from the graph and return a numpy array."""
    return x.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def train_val_test_split(
    x_data,
    y_data,
    val_split=0.1,
    test_split=0.1,
    stratify=False,
    bins=8,
    seed=None,
    device="cpu",
):
    """Three-way split into train / validation / test sets.

    Supports optional stratification via quantile binning, which preserves
    the target distribution across splits (recommended for regression with
    skewed outputs).

    Parameters
    ----------
    x_data : torch.Tensor  shape [N, n_features]
    y_data : torch.Tensor  shape [N] or [N, 1]
    val_split : float   fraction of the full dataset held out for validation
    test_split : float  fraction held out for test
    stratify : bool     if True, bin targets and stratify both splits
    bins : int          number of quantile bins (ignored when stratify=False)
    seed : int or None  random seed (None → non-reproducible)
    device : str        device for returned tensors

    Returns
    -------
    dataset : dict with keys
        "train_input", "train_label",
        "val_input",   "val_label",
        "test_input",  "test_label"
    """
    assert val_split + test_split < 1.0, "val_split + test_split must be < 1"

    X = x_data.detach().cpu().numpy()
    y = y_data.detach().cpu().numpy().squeeze()

    if stratify:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(y, quantiles))
        strat_labels = np.digitize(y, edges[1:-1])
    else:
        strat_labels = np.zeros(len(y))

    temp_fraction = val_split + test_split

    train_idx, temp_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=temp_fraction, random_state=seed)
        .split(X, strat_labels)
    )

    val_idx, test_idx = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_split / temp_fraction,
            random_state=seed,
        ).split(X[temp_idx], strat_labels[temp_idx])
    )
    val_idx  = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    def _to_tensor(arr, ref):
        return torch.as_tensor(arr, dtype=ref.dtype).to(device)

    y_np = y_data.detach().cpu().numpy()
    return {
        "train_input": _to_tensor(X[train_idx],       x_data),
        "train_label": _to_tensor(y_np[train_idx],    y_data),
        "val_input":   _to_tensor(X[val_idx],         x_data),
        "val_label":   _to_tensor(y_np[val_idx],      y_data),
        "test_input":  _to_tensor(X[test_idx],        x_data),
        "test_label":  _to_tensor(y_np[test_idx],     y_data),
    }


# ---------------------------------------------------------------------------
# KAN subclass with validation-based best-model checkpointing
# ---------------------------------------------------------------------------

class best_loss_KAN(KAN):
    """KAN subclass that saves the model state with the lowest *validation* loss.

    During training, after each step the validation loss is computed.  If it
    improves, the current state dict is deep-copied.  At the end of training
    the best state is restored, giving a form of early stopping without
    terminating training early.

    The dataset dict must include ``"val_input"`` and ``"val_label"`` in
    addition to the standard pyKAN keys.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def best_loss_fit(
        self,
        dataset,
        opt="LBFGS",
        steps=100,
        log=1,
        lamb=0.0,
        lamb_l1=1.0,
        lamb_entropy=2.0,
        lamb_coef=0.0,
        lamb_coefdiff=0.0,
        update_grid=True,
        grid_update_num=10,
        loss_fn=None,
        lr=1.0,
        start_grid_update_step=-1,
        stop_grid_update_step=50,
        batch=-1,
        metrics=None,
        save_fig=False,
        in_vars=None,
        out_vars=None,
        beta=3,
        save_fig_freq=1,
        img_folder="./video",
        singularity_avoiding=False,
        y_th=1000.0,
        reg_metric="edge_forward_spline_n",
        display_metrics=None,
        verbose=True,
    ):
        """Train the model, restoring the checkpoint with the best validation loss.

        Parameters
        ----------
        dataset : dict
            Must contain ``train_input``, ``train_label``, ``test_input``,
            ``test_label``, ``val_input``, ``val_label``.
        opt : str
            ``"LBFGS"`` (default) or ``"Adam"``.
        steps : int
            Number of training steps.
        log : int
            Progress-bar update frequency (steps).
        lamb : float
            Master regularisation weight.  The effective entropy penalty is
            ``lamb * lamb_entropy``.  See module docstring for details.
        lamb_l1 : float
            L1 penalty coefficient (relative to ``lamb``).
        lamb_entropy : float
            Entropy penalty coefficient (relative to ``lamb``).  When loading
            from a best-hyperparameter JSON, pass
            ``params["lamb_entropy"] / params["lamb"]`` here.
        loss_fn : callable or None
            Loss function ``f(pred, target) -> scalar``.  Defaults to MSE.
        verbose : bool
            If True, display a tqdm progress bar.

        Returns
        -------
        results : dict
            Training history arrays (``train_loss``, ``test_loss``,
            ``val_loss``, ``reg``, ``best_val_loss``, ``best_val_test_loss``).
        best_val_test_loss : float
            Test loss at the step with the best validation loss.
        """
        if lamb > 0.0 and not self.save_act:
            print("setting lamb=0. If you want to set lamb > 0, set self.save_act=True")

        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc="description", ncols=150) if verbose else range(steps)

        if loss_fn is None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(
                self.get_params(),
                lr=lr,
                history_size=10,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-32,
                tolerance_change=1e-32,
                tolerance_ys=1e-32,
            )

        results = {
            "train_loss": [],
            "test_loss": [],
            "val_loss": [],
            "reg": [],
            "best_val_loss": [],
            "best_val_test_loss": [],
        }
        if metrics is not None:
            for m in metrics:
                results[m.__name__] = []

        if batch == -1 or batch > dataset["train_input"].shape[0]:
            batch_size     = dataset["train_input"].shape[0]
            batch_size_val = dataset["val_input"].shape[0]
            batch_size_test = dataset["test_input"].shape[0]
        else:
            batch_size = batch
            batch_size_val = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(
                dataset["train_input"][train_id],
                singularity_avoiding=singularity_avoiding,
                y_th=y_th,
            )
            train_loss = loss_fn(pred, dataset["train_label"][train_id])
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.0)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig and not os.path.exists(img_folder):
            os.makedirs(img_folder)

        best_val_loss = 1e888
        best_test_loss = 1e888
        best_model_state = None

        for step in pbar:

            if step == steps - 1 and old_save_act:
                self.save_act = True

            if save_fig and step % save_fig_freq == 0:
                _save_act = self.save_act
                self.save_act = True

            train_id = np.random.choice(dataset["train_input"].shape[0], batch_size, replace=False)
            val_id   = np.random.choice(dataset["val_input"].shape[0],   batch_size_val,  replace=False)
            test_id  = np.random.choice(dataset["test_input"].shape[0],  batch_size_test, replace=False)

            if (
                step % grid_update_freq == 0
                and step < stop_grid_update_step
                and update_grid
                and step >= start_grid_update_step
            ):
                self.update_grid(dataset["train_input"][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(
                    dataset["train_input"][train_id],
                    singularity_avoiding=singularity_avoiding,
                    y_th=y_th,
                )
                train_loss = loss_fn(pred, dataset["train_label"][train_id])
                if self.save_act:
                    if reg_metric == "edge_backward":
                        self.attribute()
                    if reg_metric == "node_backward":
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.0)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss  = loss_fn_eval(self.forward(dataset["val_input"][val_id]),   dataset["val_label"][val_id])
            test_loss = loss_fn_eval(self.forward(dataset["test_input"][test_id]), dataset["test_label"][test_id])

            with torch.no_grad():
                if val_loss < best_val_loss:
                    best_val_loss      = val_loss.cpu().item()
                    best_val_test_loss = test_loss.cpu().item()
                    best_model_state   = copy.deepcopy(self.state_dict())
                if test_loss < best_test_loss:
                    best_test_loss = test_loss.cpu().item()

            if metrics is not None:
                for m in metrics:
                    results[m.__name__].append(m().item())

            results["train_loss"].append(train_loss.cpu().detach().numpy())
            results["test_loss"].append(test_loss.cpu().detach().numpy())
            results["val_loss"].append(val_loss.cpu().detach().numpy())
            results["reg"].append(reg_.cpu().detach().numpy())
            results["best_val_loss"].append(best_val_loss)
            results["best_val_test_loss"].append(best_val_test_loss)

            if verbose and step % log == 0:
                if display_metrics is None:
                    pbar.set_description(
                        "| train_loss: %.2e | val_loss: %.2e | reg: %.2e"
                        " | best_val_loss: %.2e | best_val_test_loss: %.2e"
                        " | best_test_loss: %.2e"
                        % (
                            train_loss.cpu().detach().numpy(),
                            val_loss.cpu().detach().numpy(),
                            reg_.cpu().detach().numpy(),
                            best_val_loss,
                            best_val_test_loss,
                            best_test_loss,
                        )
                    )
                else:
                    string, data = "", ()
                    for metric in display_metrics:
                        if metric not in results:
                            raise ValueError(f"{metric} not recognised")
                        string += f" {metric}: %.2e |"
                        data   += (results[metric][-1],)
                    pbar.set_description(string % data)

            if save_fig and step % save_fig_freq == 0:
                import matplotlib.pyplot as plt
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars,
                          title=f"Step {step}", beta=beta)
                plt.savefig(os.path.join(img_folder, f"{step}.jpg"), bbox_inches="tight", dpi=200)
                plt.close()
                self.save_act = _save_act

        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        self.log_history("fit")
        self.symbolic_enabled = old_symbolic_enabled
        return results, best_val_test_loss


# ---------------------------------------------------------------------------
# Mini-batch retraining with best-val-loss checkpointing
# ---------------------------------------------------------------------------

def mini_batch_train_best_model(
    dataset,
    model,
    epochs,
    batch_size,
    loss_fn=nn.L1Loss,
    optimiser_fn=torch.optim.Adam,
    lr=1e-3,
    scheduler_fn=None,
    penalty_fn=None,
    verbose=False,
):
    """Mini-batch Adam retraining with validation-loss checkpointing.

    Designed for use after edge pruning: retrains an existing model in-place
    and restores the parameter state that achieved the lowest validation loss,
    equivalent to the checkpointing in ``best_loss_KAN.best_loss_fit`` but
    using mini-batch SGD/Adam rather than LBFGS.

    Parameters
    ----------
    dataset : dict
        Must contain ``train_input``, ``train_label``, ``val_input``,
        ``val_label``.
    model : nn.Module
        The model to retrain in-place (e.g. a pruned ``best_loss_KAN``).
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    loss_fn : class
        Loss function *class* (not instance), e.g. ``nn.L1Loss``.
    optimiser_fn : class
        Optimiser *class* (not instance), e.g. ``torch.optim.Adam``.
    lr : float
        Learning rate.
    scheduler_fn : callable or None
        Optional LR scheduler factory ``f(optimiser) -> scheduler``.
        ``ReduceLROnPlateau`` is detected and stepped with val loss.
    penalty_fn : callable or None
        Optional extra penalty ``f(model) -> scalar``, added to the loss.
    verbose : bool
        If True, print epoch-level train and val loss.

    Returns
    -------
    best_val_loss : float
        The lowest validation loss seen during training.
    """
    lossfn    = loss_fn()
    optimiser = optimiser_fn(model.parameters(), lr)

    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimiser)

    x_train = dataset["train_input"]
    y_train = dataset["train_label"]
    x_val   = dataset["val_input"]
    y_val   = dataset["val_label"]

    n = len(x_train)
    best_val_loss      = float("inf")
    best_model_state   = None

    for epoch in range(epochs):
        model.train()
        shuffled = torch.randperm(n, device=x_train.device)

        for start in range(0, n - batch_size, batch_size):
            idx      = shuffled[start : start + batch_size]
            batch_x  = x_train[idx]
            batch_y  = y_train[idx]

            optimiser.zero_grad()
            pred = model(batch_x)
            loss = lossfn(pred.squeeze(), batch_y.squeeze())
            if penalty_fn is not None:
                loss = loss + penalty_fn(model)
            loss.backward()
            optimiser.step()

        # validation pass
        model.eval()
        with torch.no_grad():
            val_loss = lossfn(model(x_val).squeeze(), y_val.squeeze()).item()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if scheduler_fn is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if verbose:
            print(f"Epoch {epoch:>4d}  val_loss={val_loss:.4e}  best={best_val_loss:.4e}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_val_loss


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def k_fold_val(
    norm_x_data,
    norm_y_data,
    data_norm,
    model_seed=None,
    shuffle_seed=None,
    splits=5,
    n_inputs=10,
    hidden=10,
    n_outputs=1,
    grid=3,
    k=3,
    steps=25,
    loss_fn=None,
    lamb=0.0,
    lamb_entropy=0.0,
    device="cpu",
    verbose=True,
):
    """K-fold cross-validation using best_loss_KAN.

    Each held-out fold is split 50:50 into test and validation sets, and two
    models are trained per fold (A and B, swapping test/val roles).  This
    doubles the number of evaluation samples compared with a single split.

    Note on lamb_entropy
    --------------------
    Pass the *model parameter* here (i.e. ``lamb_entropy_actual / lamb``).
    When loading from a best-hyperparameter JSON file:
        ``lamb_entropy = params["lamb_entropy"] / params["lamb"]``

    Parameters
    ----------
    norm_x_data : torch.Tensor  shape [N, n_inputs]
        Normalised input data.
    norm_y_data : torch.Tensor  shape [N, 1] or [N]
        Normalised target data.
    data_norm : float
        Denormalisation scale factor (e.g. target standard deviation).
        MAE is multiplied by this before returning.
    model_seed : int or None
        Seed for KAN weight initialisation (None → random each call).
    shuffle_seed : int or None
        Seed for KFold shuffle (None → random).
    splits : int
        Number of folds.
    n_inputs : int
        Number of input features (default 10 for Berry HEA dataset).
    hidden : int
        Number of hidden nodes.
    n_outputs : int
        Number of outputs (default 1).
    grid : int
        Spline grid resolution.
    k : int
        Spline order.
    steps : int
        Training steps per model.
    loss_fn : callable or None
        Loss function. Defaults to L1Loss.
    lamb : float
        Master regularisation weight (passed directly to best_loss_fit).
    lamb_entropy : float
        Entropy penalty coefficient relative to lamb (see note above).
    device : str
        Torch device string.
    verbose : bool
        If True, print progress information.

    Returns
    -------
    mean_mae : float
        Mean absolute error across all evaluation samples (in original units).
    mean_error : float
        Standard error of the mean MAE.
    stdev_mae : float
        Sample standard deviation of per-fold MAEs.
    """
    if loss_fn is None:
        loss_fn = nn.L1Loss()

    if shuffle_seed is None:
        shuffle_seed = int(np.random.default_rng().integers(1_000_000))
    if model_seed is None:
        model_seed = int(np.random.default_rng().integers(1_000_000))

    kfold = KFold(n_splits=splits, shuffle=True, random_state=shuffle_seed)
    output_vals = []

    for train_index, test_index in kfold.split(norm_x_data):

        x_train = norm_x_data[train_index]
        y_train = norm_y_data[train_index]
        x_tv    = norm_x_data[test_index]
        y_tv    = norm_y_data[test_index]

        x_a, x_b, y_a, y_b = sklearn.model_selection.train_test_split(
            x_tv, y_tv, test_size=0.5, random_state=None, shuffle=True
        )

        for x_test, y_test, x_val, y_val in [(x_a, y_a, x_b, y_b), (x_b, y_b, x_a, y_a)]:

            dataset = {
                "train_input": x_train,
                "train_label": y_train,
                "test_input":  x_test,
                "test_label":  y_test,
                "val_input":   x_val,
                "val_label":   y_val,
            }

            model = best_loss_KAN(
                width=[n_inputs, hidden, n_outputs],
                grid=grid,
                k=k,
                device=device,
                seed=model_seed,
                auto_save=False,
            )

            model.best_loss_fit(
                dataset,
                opt="LBFGS",
                steps=steps,
                loss_fn=loss_fn,
                lamb=lamb,
                lamb_entropy=lamb_entropy,
                verbose=verbose,
            )

            with torch.no_grad():
                mae = (
                    torch.mean(torch.abs(y_test - model(x_test))).item()
                    * data_norm
                )
            output_vals.append(mae)

            del model
            torch.cuda.empty_cache()

    mean_mae  = np.mean(output_vals)
    stdev_mae = np.std(output_vals, ddof=1)
    mean_error = stdev_mae / np.sqrt(len(output_vals))
    return mean_mae, mean_error, stdev_mae


# ---------------------------------------------------------------------------
# Optuna result extraction
# ---------------------------------------------------------------------------

def get_best_result(study):
    """Extract a summary dict from the best trial of an Optuna study.

    Returns
    -------
    dict with keys: hidden, edges, meanHV, error, stddev, params
    """
    t = study.best_trial
    return {
        "hidden":  t.user_attrs["hidden"],
        "edges":   t.user_attrs["edges"],
        "meanHV":  t.value,
        "error":   t.user_attrs["error"],
        "stddev":  t.user_attrs["stddev"],
        "params":  t.params,
    }


# ---------------------------------------------------------------------------
# Edge pruning utilities
# ---------------------------------------------------------------------------

def sort_edge_scores(model):
    """Return all edge scores sorted from lowest to highest.

    Parameters
    ----------
    model : best_loss_KAN (or KAN)

    Returns
    -------
    sorted_score : torch.Tensor  [n_edges]
    sorted_score_layer : torch.Tensor  [n_edges]  layer index for each edge
    sorted_score_idx : torch.Tensor  [n_edges, 2]  (row, col) within layer
    """
    with torch.no_grad():
        model.attribute()
        scores = model.edge_scores

        score_vals_layer  = []
        score_layer       = []
        score_layer_idx   = []

        for layer_idx, layer_scores in enumerate(scores):
            flat = layer_scores.flatten()
            score_vals_layer.append(flat)
            score_layer.append(torch.full_like(flat, layer_idx))

            indices_per_dim = torch.meshgrid(
                *[torch.arange(s, device=layer_scores.device) for s in layer_scores.shape],
                indexing="ij",
            )
            flat_idx = torch.stack(indices_per_dim, dim=-1).view(-1, 2)
            score_layer_idx.append(flat_idx)

        full_score_list  = torch.tensor([v.item() for sub in score_vals_layer for v in sub])
        full_score_layer = torch.tensor([v.item() for sub in score_layer       for v in sub])
        full_score_idx   = torch.tensor([v.tolist() for sub in score_layer_idx for v in sub])

        sorted_score, sorted_indices = torch.sort(full_score_list)
        return (
            sorted_score,
            full_score_layer[sorted_indices],
            full_score_idx[sorted_indices],
        )


def remove_ranked_edges(model, n_remove=1):
    """Remove the n_remove lowest-scoring active edges from the model.

    Edges already zeroed (score == 0) are excluded from ranking so only
    truly active edges are pruned.

    Parameters
    ----------
    model : best_loss_KAN (or KAN)
    n_remove : int
        Number of edges to remove.
    """
    scores, layers, idxs = sort_edge_scores(model)

    active_mask   = scores != 0
    active_scores = scores[active_mask]
    active_layers = layers[active_mask]
    active_idxs   = idxs[active_mask]

    for layer, idx in zip(active_layers[:n_remove], active_idxs[:n_remove]):
        l = int(layer.item())
        # edge_scores is transposed relative to model.mask, so swap i/j
        i = int(idx.tolist()[1])
        j = int(idx.tolist()[0])
        model.remove_edge(l, i, j)


# ---------------------------------------------------------------------------
# Edge ablation schedule
# ---------------------------------------------------------------------------

# Default three-phase schedule developed empirically on the Berry HEA dataset
# (hidden=20, 220 initial edges).  Phases:
#   coarse      — rapid bulk removal to escape the dense regime quickly
#   intermediate — slower removal as the model approaches the sparse regime
#   fine         — one-edge-at-a-time removal near the target edge count
DEFAULT_ABLATION_SCHEDULE = [
    {"n_remove": 20, "epochs": 20, "batch_size": 128, "lr": 2e-3, "repeats": 8},
    {"n_remove": 10, "epochs": 20, "batch_size": 32,  "lr": 2e-3, "repeats": 3},
    {"n_remove":  5, "epochs": 20, "batch_size": 32,  "lr": 2e-3, "repeats": 3},
]


def ablate_edges(
    model,
    dataset,
    data_norm,
    schedule=None,
    target_edges=None,
    loss_fn=nn.L1Loss,
    optimiser_fn=torch.optim.Adam,
    verbose=True,
):
    """Iteratively prune edges and retrain, tracking MAE at each step.

    Applies a multi-phase pruning schedule: each phase removes a fixed number
    of the lowest-scoring edges, retrains with mini-batch Adam (using
    validation-loss checkpointing), then records the edge count and test MAE.
    The initial (unpruned) state is recorded as the first point.

    Parameters
    ----------
    model : best_loss_KAN (or KAN)
        A trained model to prune in-place.
    dataset : dict
        Must contain ``train_input``, ``train_label``, ``val_input``,
        ``val_label``, ``test_input``, ``test_label``.
    data_norm : float
        Denormalisation scale factor (e.g. target standard deviation) used
        to convert normalised MAE → original units.
    schedule : list of dict or None
        Pruning schedule.  Each entry must have keys:
        ``n_remove``, ``epochs``, ``batch_size``, ``lr``, ``repeats``.
        Defaults to ``DEFAULT_ABLATION_SCHEDULE`` — the three-phase routine
        (coarse → intermediate → fine) developed for the Berry HEA dataset.
    target_edges : int or None
        If set, stop pruning as soon as ``model.n_edge <= target_edges``.
        Useful for producing a model at a specific sparsity level for
        visualisation.  The final retrain is still completed before stopping.
    loss_fn : class
        Loss function class, e.g. ``nn.L1Loss``.
    optimiser_fn : class
        Optimiser class, e.g. ``torch.optim.Adam``.
    verbose : bool
        If True, print edge count and MAE after each step.

    Returns
    -------
    edge_counts : list of int
        Edge count recorded at each checkpoint (initial + after every retrain).
    maes : list of float
        Test MAE (in original units) at each checkpoint.

    Example
    -------
    Run the default schedule, then plot MAE vs edges::

        edge_counts, maes = ablate_edges(model, dataset, data_norm=stdev["HV"])

        plt.plot(edge_counts, maes, marker="o")
        plt.gca().invert_xaxis()
        plt.xlabel("Edges")
        plt.ylabel("MAE (HV)")

    Prune to a specific target for visualisation::

        ablate_edges(model, dataset, data_norm=stdev["HV"], target_edges=20)
        pruned = model.copy()
        pruned = pruned.prune(edge_th=0, node_th=0)
        pruned = pruned.prune_input(threshold=0)
        pruned.plot(scale=2)
    """
    if schedule is None:
        schedule = DEFAULT_ABLATION_SCHEDULE

    lossfn_eval = loss_fn()

    edge_counts = []
    maes        = []

    # record initial state before any pruning
    model.eval()
    with torch.no_grad():
        mae = (
            lossfn_eval(
                model(dataset["test_input"]).squeeze(),
                dataset["test_label"].squeeze(),
            ).item()
            * data_norm
        )
    edge_counts.append(model.n_edge)
    maes.append(mae)

    if verbose:
        print(f"Initial : {model.n_edge:>4d} edges  MAE = {mae:.2f}")

    done = False
    for phase_idx, phase in enumerate(schedule):
        if done:
            break

        n_remove   = phase["n_remove"]
        epochs     = phase["epochs"]
        batch_size = phase["batch_size"]
        lr         = phase["lr"]
        repeats    = phase["repeats"]

        if verbose:
            print(f"Phase {phase_idx + 1}: remove {n_remove} edges × {repeats} "
                  f"({epochs} epochs, batch={batch_size}, lr={lr})")

        for _ in range(repeats):
            remove_ranked_edges(model, n_remove=n_remove)
            mini_batch_train_best_model(
                dataset, model, epochs, batch_size,
                loss_fn=loss_fn,
                optimiser_fn=optimiser_fn,
                lr=lr,
                verbose=False,
            )

            model.eval()
            with torch.no_grad():
                mae = (
                    lossfn_eval(
                        model(dataset["test_input"]).squeeze(),
                        dataset["test_label"].squeeze(),
                    ).item()
                    * data_norm
                )
            edge_counts.append(model.n_edge)
            maes.append(mae)

            if verbose:
                print(f"  {model.n_edge:>4d} edges  MAE = {mae:.2f}")

            if target_edges is not None and model.n_edge <= target_edges:
                done = True
                break

    return edge_counts, maes
