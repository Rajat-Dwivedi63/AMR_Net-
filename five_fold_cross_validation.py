# =============================================================================
# 5_fold_cross_validation.py
# Evaluates AMRNet using 5-fold Stratified Cross-Validation on DRIAMS-A.
#
# Key Features:
#   1. StratifiedKFold (n_splits=5) preserves resistant/susceptible class ratios.
#   2. Leak-free scaling: StandardScaler is fitted strictly on each training fold.
#   3. Dynamic batch rebalancing: WeightedRandomSampler on training splits.
#   4. Early stopping with patience=15 on validation AUROC.
#   5. Exports mean ± std (AUROC, AUPRC, F1, Accuracy) for manuscript tables.
# =============================================================================

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score
)

# Mount Google Drive if in Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

# --- Configuration ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Executing on device: {DEVICE}")

DATA_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
OUTPUT_CSV = os.path.join(DATA_DIR, 'five_fold_cv_results.csv')

SPECIES_ANTIBIOTICS = [
    ('Escherichia coli', 'Ciprofloxacin'),
    ('Klebsiella pneumoniae', 'Ceftriaxone'),
    ('Staphylococcus aureus', 'Oxacillin')
]


# %% Step 1 — Dataset Class & Sampler
class SpectraDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_weighted_sampler(y_train):
    """Calculates inverse class frequencies to rebalance imbalanced batches."""
    y_int = y_train.astype(int)
    class_counts = np.bincount(y_int)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_int]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True
    )


# %% Step 2 — AMRNet Architecture
class AMRNet(nn.Module):
    def __init__(self, input_dim=6000, hidden_dims=(1024, 256, 64), dropout=0.3):
        super().__init__()
        # Input-level attention mechanism (per-bin weight gate)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        # Deep representation backbone
        layers = []
        prev_dim = input_dim
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

    def forward(self, x, return_attention=False):
        attn_weights = self.attention(x)
        features = self.backbone(x * attn_weights)
        logits = self.classifier(features).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits


# %% Step 3 — Single Fold Training & Validation Loop
def train_and_eval_fold(X_train, y_train, X_val, y_val, epochs=60, batch_size=64, lr=1e-3, patience=15):
    # Leak-free scaling: fit strictly on training fold
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    # Data loaders
    train_sampler = get_weighted_sampler(y_train)
    train_loader = DataLoader(
        SpectraDataset(X_train, y_train),
        batch_size=batch_size,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        SpectraDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    # Model initialization
    model = AMRNet(input_dim=X_train.shape[1]).to(DEVICE)
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

        # Fold validation evaluation
        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                probs = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
                val_probs.extend(probs.flatten())
                val_true.extend(yb.numpy().flatten())

        val_true = np.array(val_true)
        val_probs = np.array(val_probs)
        current_auroc = roc_auc_score(val_true, val_probs)

        # Early stopping tracking
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    val_probs, val_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            probs = torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()
            val_probs.extend(probs.flatten())
            val_true.extend(yb.numpy().flatten())

    val_true = np.array(val_true)
    val_probs = np.array(val_probs)
    val_pred = (val_probs >= 0.5).astype(int)

    metrics = {
        'auroc': roc_auc_score(val_true, val_probs),
        'auprc': average_precision_score(val_true, val_probs),
        'f1': f1_score(val_true, val_pred, zero_division=0),
        'acc': accuracy_score(val_true, val_pred)
    }

    del model, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# %% Step 4 — 5-Fold Cross-Validation Execution
summary_rows = []

for species, antibiotic in SPECIES_ANTIBIOTICS:
    tag = f"{species.replace(' ', '_')}_{antibiotic}"
    npz_path = os.path.join(DATA_DIR, f"{tag}.npz")

    if not os.path.exists(npz_path):
        print(f"Skipping {tag}: {npz_path} not found.")
        continue

    print(f"\n{'='*70}")
    print(f"Starting 5-Fold Cross-Validation: {species} / {antibiotic}")
    print(f"{'='*70}")

    data = np.load(npz_path)
    # Reassemble complete dataset to form clean cross-validation folds
    X = np.concatenate([data['X_train'], data['X_val'], data['X_test']], axis=0)
    y = np.concatenate([data['y_train'], data['y_val'], data['y_test']], axis=0)

    print(f"Total samples: {len(y)} (Resistant: {int(y.sum())}, Susceptible: {int((y == 0).sum())})")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        metrics = train_and_eval_fold(X_tr, y_tr, X_va, y_va)
        fold_metrics.append(metrics)

        print(f"  Fold {fold + 1}/5 | "
              f"AUROC: {metrics['auroc']:.4f} | "
              f"AUPRC: {metrics['auprc']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"Acc: {metrics['acc']:.4f}")

    # Compute fold statistics (mean ± std)
    df_folds = pd.DataFrame(fold_metrics)
    mean_scores = df_folds.mean()
    std_scores = df_folds.std()

    print(f"\n--- {species} / {antibiotic} Aggregate (Mean ± Std) ---")
    print(f"AUROC:    {mean_scores['auroc']:.4f} ± {std_scores['auroc']:.4f}")
    print(f"AUPRC:    {mean_scores['auprc']:.4f} ± {std_scores['auprc']:.4f}")
    print(f"F1-Score: {mean_scores['f1']:.4f} ± {std_scores['f1']:.4f}")
    print(f"Accuracy: {mean_scores['acc']:.4f} ± {std_scores['acc']:.4f}")

    summary_rows.append({
        'Species': species,
        'Antibiotic': antibiotic,
        'AUROC': f"{mean_scores['auroc']:.3f} ± {std_scores['auroc']:.3f}",
        'AUPRC': f"{mean_scores['auprc']:.3f} ± {std_scores['auprc']:.3f}",
        'F1': f"{mean_scores['f1']:.3f} ± {std_scores['f1']:.3f}",
        'Accuracy': f"{mean_scores['acc']:.3f} ± {std_scores['acc']:.3f}",
        'Raw_AUROC_Mean': mean_scores['auroc'],
        'Raw_AUROC_Std': std_scores['auroc'],
    })

# %% Step 5 — Save Summary Results Table
if summary_rows:
    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'='*70}")
    print(f"5-Fold Cross-Validation completed successfully.")
    print(f"Results table saved to: {OUTPUT_CSV}")
    print(f"{'='*70}\n")
    print(results_df[['Species', 'Antibiotic', 'AUROC', 'AUPRC', 'F1', 'Accuracy']].to_string(index=False))
