# =============================================================================
# benchmark_sota_baselines.py
#
# Comparative Benchmarking & Paired DeLong Statistical Significance Suite
# Models Evaluated:
#   1. Weis et al. (Random Forest Ensemble)
#   2. MSDeepAMR (Fully Connected Deep Neural Network Baseline)
#   3. AMRNet (Attention-Augmented Architecture - Proposed)
# =============================================================================

import os
import gc
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional Colab mount
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {DEVICE}")

DATA_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
OUTPUT_CSV = os.path.join(DATA_DIR, 'sota_benchmark_delong_results.csv')

PAIRS = [
    ('Escherichia_coli', 'Ciprofloxacin'),
    ('Klebsiella_pneumoniae', 'Ceftriaxone'),
    ('Staphylococcus_aureus', 'Oxacillin')
]


# =============================================================================
# 1. Dataset Class & Sampler
# =============================================================================
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_weighted_sampler(y):
    class_counts = np.bincount(y.astype(int))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y.astype(int)]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )


# =============================================================================
# 2. Model Architectures
# =============================================================================
class MSDeepAMR(nn.Module):
    """
    Official fully connected deep neural network architecture from
    López-Cortés et al. (Frontiers in Microbiology, 2024).
    """
    def __init__(self, input_dim=6000):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class AMRNet(nn.Module):
    """
    Proposed Attention-Augmented Architecture:
    Input attention gating preserving discrete coordinate positions,
    followed by a deep representation backbone.
    """
    def __init__(self, input_dim=6000, hidden_dims=(1024, 256, 64), dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        layers, prev_dim = [], input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x):
        attn_weights = self.attention(x)
        features = self.backbone(x * attn_weights)
        return self.classifier(features).squeeze(-1)


# =============================================================================
# 3. Training & Prediction Routines
# =============================================================================
def train_neural_model(model, X_train, y_train, X_val, y_val, epochs=60, batch_size=32, lr=1e-3, patience=15):
    sampler = get_weighted_sampler(y_train)
    train_loader = DataLoader(SpectraDataset(X_train, y_train), batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(SpectraDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_auroc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
                val_true.extend(yb.numpy())

        val_auroc = roc_auc_score(val_true, val_probs)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_proba_nn(model, X_test):
    model.eval()
    loader = DataLoader(SpectraDataset(X_test, np.zeros(len(X_test))), batch_size=64, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
    return np.array(probs)


# =============================================================================
# 4. Paired DeLong Test Implementation
# =============================================================================
def delong_roc_variance(ground_truth, predictions):
    """
    Computes structural midranks and asymptotic covariance matrix for AUC
    following DeLong et al. (Biometrics, 1988).
    """
    ground_truth = np.asarray(ground_truth)
    predictions = [np.asarray(p) for p in predictions]
    k = len(predictions)

    pos_mask = (ground_truth == 1)
    neg_mask = (ground_truth == 0)
    m = np.sum(pos_mask)
    n = np.sum(neg_mask)

    v10 = np.empty((m, k))
    v01 = np.empty((n, k))
    aucs = np.empty(k)

    for r in range(k):
        pos_scores = predictions[r][pos_mask]
        neg_scores = predictions[r][neg_mask]

        # Indicator matrix with midranks for ties
        diff = pos_scores[:, None] - neg_scores[None, :]
        psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))

        v10[:, r] = np.mean(psi, axis=1)
        v01[:, r] = 1.0 - np.mean(psi, axis=0)
        aucs[r] = np.mean(v10[:, r])

    # Sample covariance matrices
    s10 = np.cov(v10, rowvar=False) if m > 1 else np.zeros((k, k))
    s01 = np.cov(v01, rowvar=False) if n > 1 else np.zeros((k, k))

    s10 = np.atleast_2d(s10)
    s01 = np.atleast_2d(s01)

    sigma = (s10 / m) + (s01 / n)
    return aucs, sigma


def paired_delong_test(ground_truth, pred_a, pred_b):
    """
    Performs two-tailed paired DeLong test between model A and model B.
    Returns: auc_a, auc_b, z_score, p_value
    """
    aucs, sigma = delong_roc_variance(ground_truth, [pred_a, pred_b])
    diff = aucs[0] - aucs[1]
    var_diff = sigma[0, 0] + sigma[1, 1] - 2.0 * sigma[0, 1]

    if var_diff <= 0:
        return aucs[0], aucs[1], 0.0, 1.0

    z_score = diff / np.sqrt(var_diff)
    p_value = 2.0 * stats.norm.sf(np.abs(z_score))
    return aucs[0], aucs[1], z_score, p_value


# =============================================================================
# 5. Benchmark Execution Loop
# =============================================================================
benchmark_rows = []

for sp, ab in PAIRS:
    tag = f"{sp}_{ab}"
    npz_file = os.path.join(DATA_DIR, f"{tag}.npz")

    if not os.path.exists(npz_file):
        print(f"Skipping {tag}: processed file {npz_file} not found.")
        continue

    print(f"\n{'='*75}")
    print(f"Benchmarking Cohort: {sp.replace('_', ' ')} / {ab}")
    print(f"{'='*75}")

    data = np.load(npz_file)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']

    input_dim = X_train.shape[1]

    # --- Baseline 1: Weis et al. (Random Forest) ---
    print("Training Weis et al. Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, p_rf)

    # --- Baseline 2: MSDeepAMR (Dense Architecture) ---
    print("Training MSDeepAMR Baseline (López-Cortés et al.)...")
    msdeep = MSDeepAMR(input_dim=input_dim).to(DEVICE)
    msdeep = train_neural_model(msdeep, X_train, y_train, X_val, y_val, epochs=60, batch_size=32, lr=1e-3)
    p_msdeep = predict_proba_nn(msdeep, X_test)
    auc_msdeep = roc_auc_score(y_test, p_msdeep)

    # --- Model: AMRNet (Attention Gate - Proposed) ---
    print("Training AMRNet (Proposed Architecture)...")
    amrnet = AMRNet(input_dim=input_dim).to(DEVICE)
    amrnet = train_neural_model(amrnet, X_train, y_train, X_val, y_val, epochs=60, batch_size=32, lr=1e-3)
    p_amrnet = predict_proba_nn(amrnet, X_test)
    auc_amrnet = roc_auc_score(y_test, p_amrnet)

    # --- Paired DeLong Statistical Tests ---
    _, _, z_vs_rf, p_vs_rf = paired_delong_test(y_test, p_amrnet, p_rf)
    _, _, z_vs_msdeep, p_vs_msdeep = paired_delong_test(y_test, p_amrnet, p_msdeep)

    cohort_label = f"{sp.replace('_', ' ')} / {ab}"
    print(f"Results for {cohort_label}:")
    print(f"  Random Forest AUROC : {auc_rf:.4f}")
    print(f"  MSDeepAMR     AUROC : {auc_msdeep:.4f}")
    print(f"  AMRNet (Ours) AUROC : {auc_amrnet:.4f}")
    print(f"  DeLong Test vs RF   : z = {z_vs_rf:.2f}, p = {p_vs_rf:.4e}")
    print(f"  DeLong Test vs MSDeep: z = {z_vs_msdeep:.2f}, p = {p_vs_msdeep:.4e}")

    benchmark_rows.append({
        'Species / Antibiotic': cohort_label,
        'Weis et al. (RF)': round(auc_rf, 4),
        'MSDeepAMR (Repro.)': round(auc_msdeep, 4),
        'AMRNet (Ours)': round(auc_amrnet, 4),
        'DeLong p (vs. RF)': f"{p_vs_rf:.4f}" if p_vs_rf >= 1e-4 else f"{p_vs_rf:.2e}",
        'DeLong p (vs. MSDeep)': f"{p_vs_msdeep:.4f}" if p_vs_msdeep >= 1e-4 else f"{p_vs_msdeep:.2e}"
    })

    # Free memory
    del rf, msdeep, amrnet, data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# 6. Save & Display Final Table
# =============================================================================
df_results = pd.DataFrame(benchmark_rows)
df_results.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("TABLE V: COMPARATIVE SOTA BENCHMARK & PAIRED DELONG SIGNIFICANCE TESTS")
print("=" * 80)
print(df_results.to_string(index=False))
print("=" * 80)
print(f"Results saved to: {OUTPUT_CSV}\n")
