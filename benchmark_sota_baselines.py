# =============================================================================
# benchmark_sota_baselines.py
# Evaluates Weis et al. (RF) and 1D-CNN against AMRNet with DeLong's test
# =============================================================================

import os
import gc
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Mount Google Drive if running in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

DATA_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
OUTPUT_CSV = os.path.join(DATA_DIR, 'sota_benchmark_comparison.csv')

PAIRS = [
    ('Escherichia_coli', 'Ciprofloxacin'),
    ('Klebsiella_pneumoniae', 'Ceftriaxone'),
    ('Staphylococcus_aureus', 'Oxacillin')
]

# --- 1. Dataset & Architectures ---
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class AMRNet(nn.Module):
    def __init__(self, input_dim=6000, hidden_dims=(1024, 256, 64), dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())
        layers, prev_dim = [], input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x):
        attn_weights = self.attention(x)
        return self.classifier(self.backbone(x * attn_weights)).squeeze(-1)

class MALDI_1DCNN_SOTA(nn.Module):
    """1D-CNN baseline (Frontiers in Microbiology 2024 comparator)"""
    def __init__(self, in_features=6000, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.classifier(self.features(x)).squeeze(-1)

# --- 2. DeLong Test Implementation ---
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    delongcov = (np.cov(v01) / m) + (np.cov(v10) / n)
    return aucs, delongcov

def compare_delong(y_true, p_target, p_comp):
    y_true = np.array(y_true).astype(bool)
    order = np.argsort(y_true)[::-1]
    m = np.sum(y_true)
    preds = np.vstack([p_target, p_comp])[:, order]
    aucs, cov = fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(max(var_diff, 1e-12))
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p_val

# --- 3. Training & Benchmark Execution ---
results = []

for sp, ab in PAIRS:
    tag = f"{sp}_{ab}"
    npz_file = os.path.join(DATA_DIR, f"{tag}.npz")
    if not os.path.exists(npz_file):
        print(f"Skipping {tag}: {npz_file} missing.")
        continue

    print(f"\n{'='*65}\nEvaluating: {sp} / {ab}\n{'='*65}")
    data = np.load(npz_file)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'], data['y_val']
    X_te, y_te = data['X_test'], data['y_test']

    # 1. Load Trained AMRNet
    amrnet = AMRNet(input_dim=X_tr.shape[1]).to(DEVICE)
    model_path = os.path.join(DATA_DIR, f"{tag}_model.pt")
    amrnet.load_state_dict(torch.load(model_path, map_location=DEVICE))
    amrnet.eval()
    with torch.no_grad():
        p_amrnet = torch.sigmoid(amrnet(torch.tensor(X_te, dtype=torch.float32).to(DEVICE))).cpu().numpy().flatten()

    # 2. Train SOTA 1: Weis et al. Random Forest
    print("  Fitting SOTA 1: Random Forest (Weis et al.)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict_proba(X_te)[:, 1]

    # 3. Train SOTA 2: 1D-CNN (Frontiers 2024)
    print("  Training SOTA 2: 1D-CNN baseline...")
    cnn = MALDI_1DCNN_SOTA(in_features=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(SpectraDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    va_loader = DataLoader(SpectraDataset(X_va, y_va), batch_size=64)

    best_loss = float('inf')
    best_cnn_weights = None
    for ep in range(40):
        cnn.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(cnn(xb), yb).backward()
            opt.step()

        cnn.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                va_losses.append(crit(cnn(xb), yb).item())
        mean_va_loss = np.mean(va_losses)
        if mean_va_loss < best_loss:
            best_loss = mean_va_loss
            best_cnn_weights = cnn.state_dict()

    cnn.load_state_dict(best_cnn_weights)
    cnn.eval()
    with torch.no_grad():
        p_cnn = torch.sigmoid(cnn(torch.tensor(X_te, dtype=torch.float32).to(DEVICE))).cpu().numpy().flatten()

    # 4. Statistical Significance (DeLong)
    _, _, z_rf, p_delong_rf = compare_delong(y_te, p_amrnet, p_rf)
    _, _, z_cnn, p_delong_cnn = compare_delong(y_te, p_amrnet, p_cnn)

    results.append({
        'Species': sp.replace('_', ' '),
        'Antibiotic': ab,
        'AMRNet_AUROC': roc_auc_score(y_te, p_amrnet),
        'AMRNet_AUPRC': average_precision_score(y_te, p_amrnet),
        'Weis_RF_AUROC': roc_auc_score(y_te, p_rf),
        'Weis_RF_AUPRC': average_precision_score(y_te, p_rf),
        '1D_CNN_AUROC': roc_auc_score(y_te, p_cnn),
        '1D_CNN_AUPRC': average_precision_score(y_te, p_cnn),
        'p_val_vs_RF': p_delong_rf,
        'p_val_vs_CNN': p_delong_cnn
    })

    print(f"  AUROC -> AMRNet: {roc_auc_score(y_te, p_amrnet):.4f} | RF: {roc_auc_score(y_te, p_rf):.4f} | 1D-CNN: {roc_auc_score(y_te, p_cnn):.4f}")
    print(f"  DeLong p-value vs RF:  {p_delong_rf:.4e} {'(p < 0.001)' if p_delong_rf < 0.001 else ''}")
    print(f"  DeLong p-value vs CNN: {p_delong_cnn:.4e} {'(p < 0.001)' if p_delong_cnn < 0.001 else ''}")

    del cnn, rf, amrnet
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save final comparison table
summary_df = pd.DataFrame(results)
summary_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nAll benchmark evaluations complete. Summary written to {OUTPUT_CSV}")
