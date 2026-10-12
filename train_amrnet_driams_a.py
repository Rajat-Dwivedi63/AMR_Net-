# =============================================================================
# Full Pipeline for COLAB 
#Step 1: Extracting driams.zip to Colab SSD.
#Steps 2–5: Metadata coalescing, memory-safe float32 spectra extraction, label parsing, and generating .npz files.
#Steps 6–7: The AMRNet architecture and the single-split training loop.
# =============================================================================

import os
import re
import gc
import glob
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

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
print(f"Using device: {DEVICE}")

TARGET_SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Staphylococcus aureus']
SPECIES_ANTIBIOTICS = {
    'Escherichia coli': ['Ciprofloxacin'],
    'Klebsiella pneumoniae': ['Ceftriaxone'],
    'Staphylococcus aureus': ['Oxacillin'],
}

WORK_DIR = '/content/DRIAMS'
SAVE_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
ZIP_PATH = '/content/drive/MyDrive/driams.zip'
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# 1. Dataset Extraction
# =============================================================================
if not os.path.exists(os.path.join(WORK_DIR, 'DRIAMS-A')):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(WORK_DIR)
    print("Extraction complete.")
else:
    print("Dataset already extracted, skipping.")


# =============================================================================
# 2. Metadata Parsing and Coalescing
# =============================================================================
id_path = os.path.join(WORK_DIR, 'DRIAMS-A', 'id')
id_files = glob.glob(os.path.join(id_path, '**', '*.csv'), recursive=True)
print(f"Found {len(id_files)} metadata files")

meta = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in id_files], ignore_index=True
)

def coalesce_value(series):
    non_missing = series[series.astype(str) != '-']
    return non_missing.iloc[0] if len(non_missing) > 0 else series.iloc[0]

other_cols = [c for c in meta.columns if c != 'code']
meta = meta.groupby('code', as_index=False).agg(
    {c: coalesce_value for c in other_cols}
)

meta_filtered = meta[meta['species'].isin(TARGET_SPECIES)].copy()
print(meta_filtered['species'].value_counts())

del meta
gc.collect()


# =============================================================================
# 3. Memory-Mapped Spectra Extraction
# =============================================================================
binned_path = os.path.join(WORK_DIR, 'DRIAMS-A', 'binned_6000')
binned_files = glob.glob(os.path.join(binned_path, '**', '*.txt'), recursive=True)
code_to_file = {os.path.splitext(os.path.basename(f))[0]: f for f in binned_files}

matched_codes = [c for c in meta_filtered['code'] if c in code_to_file]
print(f"{len(matched_codes)} / {len(meta_filtered)} metadata records matched with spectral files.")

N_BINS = 6000
n_samples = len(matched_codes)
X_spectra = np.empty((n_samples, N_BINS), dtype=np.float32)

# Linear m/z axis: 6,000 bins, 3 Da width, spanning 2000-20000 Da
mz_axis = 2000.0 + np.arange(N_BINS, dtype=np.float32) * 3.0

for i, code in enumerate(matched_codes):
    fpath = code_to_file[code]
    arr = np.loadtxt(fpath, skiprows=1, dtype=np.float32)
    X_spectra[i, :] = arr[:, 1]
    if i % 2000 == 0:
        print(f"  Loaded {i}/{n_samples}")

np.save(os.path.join(SAVE_DIR, 'X_spectra.npy'), X_spectra)
np.save(os.path.join(SAVE_DIR, 'spectra_codes.npy'), np.array(matched_codes))
np.save(os.path.join(SAVE_DIR, 'mz_axis.npy'), mz_axis)

meta_matched = meta_filtered[meta_filtered['code'].isin(matched_codes)].copy()
meta_matched = meta_matched.set_index('code').loc[matched_codes].reset_index()
meta_matched.to_csv(os.path.join(SAVE_DIR, 'meta_matched.csv'), index=False)

del meta_filtered, code_to_file, binned_files
gc.collect()


# =============================================================================
# 4. Phenotypic Label Parsing & Split Generation
# =============================================================================
def parse_label(raw_value):
    if pd.isna(raw_value) or raw_value == '-':
        return None
    val = str(raw_value).strip()
    if val == 'R':
        return 1
    if val == 'S':
        return 0
    if val == 'I':
        return None
    tokens = re.findall(r'([RSI])\(\d+\)', val)
    if not tokens:
        return None
    if 'R' in tokens:
        return 1
    if 'S' in tokens:
        return 0
    return None

X_spectra = np.load(os.path.join(SAVE_DIR, 'X_spectra.npy'))
meta_matched = pd.read_csv(os.path.join(SAVE_DIR, 'meta_matched.csv'), low_memory=False)

codes_path = os.path.join(SAVE_DIR, 'spectra_codes.npy')
if os.path.exists(codes_path):
    spectra_codes = np.load(codes_path, allow_pickle=True)
    code_to_row = {c: i for i, c in enumerate(spectra_codes)}

    other_cols = [c for c in meta_matched.columns if c != 'code']
    meta_matched = meta_matched.groupby('code', as_index=False).agg(
        {c: coalesce_value for c in other_cols}
    )

    common_codes = [c for c in meta_matched['code'] if c in code_to_row]
    meta_matched = meta_matched.set_index('code').loc[common_codes].reset_index()
    row_order = [code_to_row[c] for c in common_codes]
    X_spectra = X_spectra[row_order]

assert len(meta_matched) == X_spectra.shape[0], "Metadata and spectra count mismatch."

def build_and_save_dataset(species, antibiotic, X_spectra, meta_matched, save_dir):
    meta_matched = meta_matched.reset_index(drop=True)
    mask = (meta_matched['species'] == species).values
    species_positions = np.where(mask)[0]

    labels = meta_matched.loc[species_positions, antibiotic].apply(parse_label)
    valid_mask = labels.notna().values
    idx = species_positions[valid_mask]

    X = X_spectra[idx]
    y = labels[valid_mask].astype(int).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    tag = f"{species.replace(' ', '_')}_{antibiotic}"
    np.savez(
        os.path.join(save_dir, f'{tag}.npz'),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    print(f"Saved {tag}.npz (Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)})")

for species, antibiotics in SPECIES_ANTIBIOTICS.items():
    for ab in antibiotics:
        build_and_save_dataset(species, ab, X_spectra, meta_matched, SAVE_DIR)

del X_spectra, meta_matched
gc.collect()


# =============================================================================
# 5. AMRNet Architecture & PyTorch Dataset
# =============================================================================
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
            layers += [nn.Linear(prev_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 1)

    def forward(self, x, return_attention=False):
        attn_weights = self.attention(x)
        features = self.backbone(x * attn_weights)
        logits = self.classifier(features).squeeze(-1)
        return (logits, attn_weights) if return_attention else logits


# =============================================================================
# 6. Training & Evaluation Pipeline
# =============================================================================
def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=1e-3):
    train_loader = DataLoader(SpectraDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SpectraDataset(X_val, y_val), batch_size=batch_size)
    model = AMRNet(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    best_auroc, best_state = 0.0, None
    patience, no_improve = 15, 0

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
            best_auroc, best_state, no_improve = val_auroc, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auroc

def evaluate_model(model, X_test, y_test):
    model.eval()
    loader = DataLoader(SpectraDataset(X_test, y_test), batch_size=64)
    probs, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            true.extend(yb.numpy())
    preds = (np.array(probs) >= 0.5).astype(int)
    return {
        'AUROC': roc_auc_score(true, probs),
        'F1': f1_score(true, preds),
        'Accuracy': accuracy_score(true, preds)
    }

results = {}
for species, antibiotics in SPECIES_ANTIBIOTICS.items():
    for ab in antibiotics:
        tag = f"{species.replace(' ', '_')}_{ab}"
        npz_path = os.path.join(SAVE_DIR, f'{tag}.npz')
        if not os.path.exists(npz_path):
            continue

        data = np.load(npz_path)
        print(f"\nTraining AMRNet: {species} / {ab}")
        model, _ = train_model(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
        metrics = evaluate_model(model, data['X_test'], data['y_test'])
        results[(species, ab)] = metrics
        print(f"Results: {metrics}")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{tag}_model.pt'))
        del model, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

results_df = pd.DataFrame(results).T
results_df.index.names = ['species', 'antibiotic']
print("\nBenchmark Summary:")
print(results_df)
results_df.to_csv(os.path.join(SAVE_DIR, 'amrnet_pipeline_results.csv'))
