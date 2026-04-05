"""
srf_experiment.py
_________________
SRF-Mal experiment module - added by Jessmon & Alwan, University of Galway (2026).
Extends Rey et al. (2022) with:
  - Krum and Bulyan aggregation
  - Label flip, gradient noise, sign flip attacks
  - Scalable cliemt simulation (10-80 clients)
  - Automatic sweep of all strategy x attack combination

This is the primary new file added to Rey et al.'s codebase.
Only main.py was minimally modified to add the --srf argument parser and route to this module.
All other original files (data.py, federated_util.py, architectures.py, etc.) are unchanged.

Called from main.py via:
    python src/main.py --srf
    python src/main.py --srf --all
    python src/main.py --srf --quick
    python src/main.py --srf --strategy krum --attack label_flip
"""

import os
import csv
import copy
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader

# -- Import Rey et al.'s aggregation functions from federated_util.py ----------
from federated_util import (federated_averaging,
                            federated_median,
                            federated_trimmed_mean_1 as federated_trimmed_mean)

INPUT_DIM = 115
SEED      = 42

#---------------------------------------------------------------------------------
# DATA LOADING
# Uses same data_path as Rey et al.'s data.py
#---------------------------------------------------------------------------------

def load_flat_dataset(data_path, sample_size=5000):
    """
    Load N-BaIoT CSVs from flat folder.
    Supports naming: 1.benign.csv, 1.mirai.ack.csv, 1.gafgyt.combo.csv
    Falls back to synthetic data if no CSVs found.
    """
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    if not csv_files:
        print("[SRF] No CSVs found - using synthetic data.")
        return _make_synthetic()
    
    dfs = []
    for f in csv_files:
        name = os.path.basename(f).lower()
        if "benign" in name:
            label = 0
        elif "mirai" in name or "gafgyt" in name:
            label = 1
        else:
            continue
        df = pd.read_csv(f, low_memory=False)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=SEED)
        df = df.copy()
        df["label"] = label
        dfs.append(df)

    if not dfs:
        print("[SRF] No valid files - using synthetic data.")
        return _make_synthetic()
    
    combined = pd.concat(dfs, ignore_index=True)
    y = combined["label"].values.astype(np.float32)
    X = combined.drop(columns=["label"]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean()).values.astype(np.float32)
    print(f"[SRF] Loaded {len(y):,} samples - "
          f"benign={int((y==0).sum()):,}, malware={int((y==1).sum()):,}")
    return X, y


def _make_synthetic():
    rng = np.random.RandomState(SEED)
    X0 = rng.randn(25000, INPUT_DIM).astype(np.float32)
    X1 = (rng.randn(25000, INPUT_DIM) + 1.5).astype(np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0]*25000 + [1]*25000, dtype=np.float32)
    idx = rng.permutation(len(y))
    print(f"[SRF] Synthetic: {len(y):,} samples, {INPUT_DIM} features.")
    return X[idx], y[idx]


def prepare_data(data_path, num_clients, alpha=0.5, sample_size=5000):
    """Load, split, normalise, partition info clients."""
    X, y = load_flat_dataset(data_path, sample_size=sample_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # Fit scaler on train only - prevents data leakege
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clients = _dirichlet_split(X_train, y_train, num_clients, alpha)
    return clients, X_test, y_test


def _dirichlet_split(X, y, num_clients, alpha):
    """Non-IID Dirichlet partition - simulates heterogeneous IoT devices."""
    rng     = np.random.RandomState(SEED)
    indices = [[] for _ in range(num_clients)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        props = rng.dirichlet(np.repeat(alpha, num_clients))
        props = np.maximum(props, 1e-3); props /= props.sum()
        splits = (props * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        pos = 0
        for i, n in enumerate(splits):
            indices[i].extend(idx[pos:pos+n].tolist()); pos += n
    clients = []
    for i in range(num_clients):
        idx = np.array(indices[i]); rng.shuffle(idx)
        clients.append((X[idx], y[idx]))
    sizes = [len(c[0]) for c in clients]
    print(f"[SRF] Non-IID split (α = {alpha}):"
          f"min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.0f}")
    return clients


#--------------------------------------------------------------------------
# MODEL
#--------------------------------------------------------------------------

class SRFMalMLP(nn.Module):
    """Simple MLP matching Rey et al.'s classifier architecture."""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
    
        )
    def forward(self, x):
        return self.net(x)
    

def _get_params(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]

def _set_params(model, params):
    sd = OrderedDict({k: torch.tensor(v)
                      for k, v in zip(model.state_dict().keys(), params)})
    model.load_state_dict(sd)

def _make_loader(X, y, batch_size=32, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



#-------------------------------------------------------------------------------
# AGGREGATION
# Uses Rey et al.'s original functions + adds Krum and Bulyan
#-------------------------------------------------------------------------------

def _fedavg(all_params):
    """Calls Rey et al.'s federated_averaging via wrapper."""
    n = len(all_params)
    return [sum(p[i] for p in all_params) / n
            for i in range(len(all_params[0]))]

def _median(all_params):
    return [np.median(np.stack([p[i] for p in all_params], axis=0), axis=0)
            for i in range (len(all_params[0]))]

def _trimmed_mean(all_params, trim=0.2):
    n = len(all_params); k = int(n * trim)
    result = []
    for i in range(len(all_params[0])):
        stack   = np.sort(np.stack([p[i] for p in all_params], axis=0), axis=0)
        trimmed = stack[k: n-k] if 2*k < n else stack
        result.append(np.mean(trimmed, axis=0))
    return result

def _krum(all_params):
    """
    Krum - Blanchard et al. (2017).
    Added by SRF-Mal as Rey et al. future work recommendation.
    """
    n = len(all_params); m = max(1, n-2)
    vecs = [np.concatenate([w.flatten() for w in p]) for p in all_params]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum((vecs[i]-vecs[j])**2)
            dist[i,j] = dist[j,i] = d
    scores = [np.sort(dist[i])[1:m+1].sum() for i in range(n)]
    return all_params[int(np.argmin(scores))]

def _bulyan(all_params):
    """
    Bulyan - El Mhamdi et al. (2018).
    Added by SRF-Mal as Rey et al. future work rcommendation.
    """
    n = len(all_params); c = max(3, n//2); m = max(1, n-2)
    vecs = [np.concatenate([w.flatten() for w in p]) for p in all_params]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range (i+1, n):
            d = np.sum((vecs[i]-vecs[j])**2)
            dist[i,j] = dist[j,i] = d
    scores    = [np.sort(dist[i])[1:m+1].sum() for i in range(n)]
    selected  = [all_params[i] for i in sorted(range(n), key=lambda i: scores[i])[:c]]
    return _trimmed_mean(selected)

AGG = {
    "fedavg":            _fedavg,
    "median":            _median,
    "trimmed_mean":      _trimmed_mean,
    "krum":              _krum,
    "bulyan":            _bulyan,
}

#------------------------------------------------------------------------------------
# ATTACKS
# Added by SRF-Mal - extends Rey et al.'s attack framework
#------------------------------------------------------------------------------------

def _label_flip(y):
    """Label Flip - data level. Flips labels 0↔1 before training."""
    return 1.0 - y

def _gradient_noise(params, std=1.0):
    """Gradient Noise - weight level. Adds gaussian noise after training."""
    rng = np.random.RandomState()
    return [p + rng.normal(0, std, p.shape).astype(p.dtype) for p in params]

def _sign_flip(params):
    """Sign Flip - weight level. Negates all weighs after training."""
    return [-p for p in params]



#------------------------------------------------------------------------------------
# LOCAL TRAIN & EVALUATE
#------------------------------------------------------------------------------------

def _local_train(model, X, y, epochs=10, lr =0.001):
    model.train()
    loader = _make_loader(X, y)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for Xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(Xb), yb).backward()
            opt.step()

def _evaluate(model, X, y):
    model.eval()
    loader = _make_loader(X, y, shuffle=False)
    probs, targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = torch.sigmoid(model(Xb)).numpy().flatten()
            probs.extend(p); targets.extend(yb.numpy().flatten())
    preds = [1 if p >= 0.5 else 0 for p in probs]
    f1 = round(f1_score(targets, preds, zero_division=0), 4)
    try:    auc = round(roc_auc_score(targets, probs), 4)
    except: auc = 0.0
    return f1, auc


#------------------------------------------------------------------------------------
# MAIN FL LOOP
#------------------------------------------------------------------------------------

def run_srf_experiment(clients, X_test, y_test,
                       strategy, attack,
                       num_rounds, local_epochs, lr,
                       malicious_clients, noise_std,
                       results_dir):
    """
    Run one FL experiment - extends Rey et al.'s fedavg_classifiers_train_test
    with robust aggregation and Byzantine attacks.
    """
    
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{strategy}_{attack}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "f1", "auc"])

    print(f"\n{'='*52}")
    print(f"  Strategy : {strategy}")
    print(f"  Attack : {attack}")
    print(f"  Clients : {len(clients)} ({len(malicious_clients)} malicious)")
    print(f"  Rounds : {num_rounds}")
    print(f"{'='*52}")

    input_dim      = clients[0][0].shape[1]
    agg_fn         = AGG[strategy]
    global_model   = SRFMalMLP(input_dim)
    global_params  = _get_params(global_model)

    for rnd in range(1, num_rounds + 1):
        all_params = []

        for cid, (X_c, y_c) in enumerate(clients):
            local_model = copy.deepcopy(global_model)
            _set_params(local_model, global_params)

            # Data-level attack: label flip before training
            if cid in malicious_clients and attack =="label_flip":
                y_c = _label_flip(y_c)

            _local_train(local_model, X_c, y_c, epochs=local_epochs, lr=lr)
            params = _get_params(local_model)

            # Weight-level attacks: after training
            if cid in malicious_clients:
                if attack == "gradient_noise":
                    params = _gradient_noise(params, std=noise_std)
                elif attack == "sign_flip":
                    params = _sign_flip(params)

            all_params.append(params)

        # Server aggregation
        global_params = agg_fn(all_params)
        _set_params(global_model, global_params)

        # Evaluate
        f1, auc = _evaluate(global_model, X_test, y_test)
        print(f" Round {rnd:>3}/{num_rounds} | F1={f1:.4f} | AUC={auc:.4f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, f1, auc])

    print(f" ✓ Saved → {csv_path}\n")


#------------------------------------------------------------------------------------
# ENTRY POINT - called from main.py
#------------------------------------------------------------------------------------

STRATEGIES = ["fedavg", "median", "trimmed_mean", "krum", "bulyan"]
ATTACKS    = ["none", "label_flip", "gradient_noise", "sign_flip"]

def run_srf(data_path, strategy="fedavg", attack="none",
            run_all=False, quick=False,
            num_clients=10, num_rounds=30,
            malicious_pct=40, sample_size=5000,
            results_dir="results"):
    """
    Main entry point for SRF-Mal experiments.
    Called from main.py --srf flag.
    """

    # Set malicious clients
    n_mal = max(1, int(num_clients * malicious_pct / 100))
    malicious_clients = list(range(n_mal))

    print(f"\n[SRF-Mal] Scalable Robust FL for IoT Malware Detection")
    print(f"[SRF-Mal] Extending Rey et al. (2022) - Jessmon & Alwan, 2026")
    print(f"[SRF-Mal] Clients={num_clients} | Malicious={n_mal} ({malicious_pct}%) | Rounds={num_rounds}")


    # Load data once
    clients, X_test, y_test = prepare_data(
        data_path, num_clients=num_clients, sample_size=sample_size)
    
    # Decide combinations
    if quick:
        combos = [("fedavg","none"), ("fedavg","label_flip"),
                  ("krum", "none"), ("krum","label_flip")]
        num_rounds = 10
    elif run_all:
        combos = [(s, a) for s in STRATEGIES for a in ATTACKS]
    else:
        combos = [(strategy, attack)]

    total = len(combos)
    for idx, (s, a) in enumerate(combos, 1):
        print(f"\n[{idx}/{total}] strategy={s} attack={a}")
        run_srf_experiment(
            clients               = clients,
            X_test                = X_test,
            y_test                = y_test,
            strategy              = s,
            attack                = a,
            num_rounds            = num_rounds,
            local_epochs          = 10,
            lr                    = 0.001,
            malicious_clients     = malicious_clients,
            noise_std             = 1.0,
            results_dir           = results_dir,
        )

    print(f"\n ✓ All done! Results saved to '{results_dir}/'")



