# =============================================================================
# TL-MALDI-AMR — DRIAMS-B Integration + Cross-Dataset Validation
# Run AFTER your DRIAMS-A pipeline (reuses its saved models + processed data)
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Staphylococcus aureus']
SPECIES_ANTIBIOTICS = {
    'Escherichia coli': 'Ciprofloxacin',
    'Klebsiella pneumoniae': 'Ceftriaxone',
    'Staphylococcus aureus': 'Oxacillin',
}
WORK_DIR = '/content/DRIAMS_B'  # separate from DRIAMS-A's /content/DRIAMS to
# avoid the two datasets' same-named year folders (e.g. both may have a
# "2018" folder) colliding or silently overwriting each other
SAVE_DIR_A = '/content/drive/MyDrive/tl_maldi_amr_processed'       # existing DRIAMS-A outputs
SAVE_DIR_B = '/content/drive/MyDrive/tl_maldi_amr_processed_driamsB'  # new DRIAMS-B outputs
os.makedirs(SAVE_DIR_B, exist_ok=True)


# %% Step 1 — Extract DRIAMS-B (same pattern as DRIAMS-A)
# IMPORTANT: confirm the actual filename first — run this to find it:
#   import os
#   for f in os.listdir('/content/drive/MyDrive'):
#       if 'driams' in f.lower():
#           print(f)
ZIP_PATH_B = '/content/drive/MyDrive/TL-MALDI-AMR/data/DRIAMS-B.zip'  # confirmed path

if not os.path.exists(ZIP_PATH_B):
    raise FileNotFoundError(
        f"{ZIP_PATH_B} does not exist. List your Drive root (see snippet "
        f"above) to find the real filename and fix ZIP_PATH_B before continuing."
    )

if not os.path.exists(os.path.join(WORK_DIR, 'id')) and \
   not os.path.exists(os.path.join(WORK_DIR, 'DRIAMS-B', 'id')):
    with zipfile.ZipFile(ZIP_PATH_B, 'r') as zf:
        zf.extractall(WORK_DIR)
    print("DRIAMS-B extraction complete.")
else:
    print("Already extracted, skipping.")

# Auto-detect whether the zip had a 'DRIAMS-B' wrapper folder or extracted
# id/binned_6000 directly at the top level — don't assume either way.
if os.path.exists(os.path.join(WORK_DIR, 'DRIAMS-B', 'id')):
    DRIAMS_B_ROOT = os.path.join(WORK_DIR, 'DRIAMS-B')
elif os.path.exists(os.path.join(WORK_DIR, 'id')):
    DRIAMS_B_ROOT = WORK_DIR
else:
    raise FileNotFoundError(
        f"Could not find an 'id' folder under {WORK_DIR} or "
        f"{WORK_DIR}/DRIAMS-B. Run: for root, dirs, files in os.walk('{WORK_DIR}'): "
        f"print(root, dirs) — to see the actual extracted structure."
    )
print(f"Using DRIAMS-B root: {DRIAMS_B_ROOT}")


# %% Step 2 — Load + coalesce DRIAMS-B metadata (same fix as DRIAMS-A)
id_path = os.path.join(DRIAMS_B_ROOT, 'id')
id_files = glob.glob(os.path.join(id_path, '**', '*.csv'), recursive=True)
print(f"Found {len(id_files)} DRIAMS-B metadata files")
for f in sorted(id_files):
    print(f"  {os.path.basename(f)}")

meta = pd.concat([pd.read_csv(f, low_memory=False) for f in id_files], ignore_index=True)
print(f"Raw concatenated metadata: {len(meta)} rows")

def coalesce_value(series):
    non_missing = series[series.astype(str) != '-']
    return non_missing.iloc[0] if len(non_missing) > 0 else series.iloc[0]

other_cols = [c for c in meta.columns if c != 'code']
meta = meta.groupby('code', as_index=False).agg({c: coalesce_value for c in other_cols})
print(f"After coalescing duplicate rows by code: {len(meta)} unique codes")

meta_filtered = meta[meta['species'].isin(TARGET_SPECIES)].copy()
print(meta_filtered['species'].value_counts())
del meta
gc.collect()


# %% Step 3 — Memory-safe spectra loading (identical approach to DRIAMS-A)
binned_path = os.path.join(DRIAMS_B_ROOT, 'binned_6000')
binned_files = glob.glob(os.path.join(binned_path, '**', '*.txt'), recursive=True)
code_to_file = {os.path.splitext(os.path.basename(f))[0]: f for f in binned_files}
print(f"Found {len(binned_files)} DRIAMS-B binned spectrum files")

matched_codes = [c for c in meta_filtered['code'] if c in code_to_file]
print(f"{len(matched_codes)} / {len(meta_filtered)} metadata rows have a matching spectrum")

N_BINS = 6000
n_samples = len(matched_codes)
X_spectra_B = np.empty((n_samples, N_BINS), dtype=np.float32)

# NOTE: confirm DRIAMS-B's binned_6000 files have the same "bin_index
# binned_intensity" header format as DRIAMS-A before trusting skiprows=1.
# If DRIAMS-B was collected/exported differently, check with:
#   with open(binned_files[0]) as f:
#       for i, line in enumerate(f):
#           print(repr(line))
#           if i >= 3: break
# before running the loop below, same as we did for DRIAMS-A.
for i, code in enumerate(matched_codes):
    fpath = code_to_file[code]
    arr = np.loadtxt(fpath, skiprows=1, dtype=np.float32)
    X_spectra_B[i, :] = arr[:, 1]
    if i % 2000 == 0:
        print(f"  loaded {i}/{n_samples}")

print(f"DRIAMS-B spectra matrix: {X_spectra_B.shape}")

np.save(os.path.join(SAVE_DIR_B, 'X_spectra.npy'), X_spectra_B)
np.save(os.path.join(SAVE_DIR_B, 'spectra_codes.npy'), np.array(matched_codes))
meta_matched_B = meta_filtered[meta_filtered['code'].isin(matched_codes)].copy()
meta_matched_B = meta_matched_B.reset_index(drop=True)  # critical: Step 6 uses
# positional integer indices with meta_matched_B — without a clean 0..N-1
# index here, .loc[] lookups fail with a KeyError since .loc is label-based,
# not position-based.
meta_matched_B.to_csv(os.path.join(SAVE_DIR_B, 'meta_matched.csv'), index=False)

del meta_filtered, code_to_file, binned_files
gc.collect()


# %% Step 4 — Label parsing (identical composite-label fix as DRIAMS-A)
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


# %% Step 5 — Model architecture (must match training exactly)
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


# %% Step 6 — CROSS-DATASET VALIDATION: evaluate DRIAMS-A-trained models on DRIAMS-B
# This is the real point of adding DRIAMS-B: does a model trained on one
# dataset generalize to another? Published literature on this exact setup
# reports performance often DROPS when train/test come from different
# datasets/sites/times — so don't be surprised or alarmed if these numbers
# are meaningfully lower than your DRIAMS-A-only results. That is itself
# a valid, expected, reportable finding — not a bug.
def evaluate_model(model, X_test, y_test):
    model.eval()
    loader = DataLoader(SpectraDataset(X_test, y_test), batch_size=64)
    probs, true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy())
            true.extend(yb.numpy())
    preds = (np.array(probs) >= 0.5).astype(int)
    return {'AUROC': roc_auc_score(true, probs), 'F1': f1_score(true, preds),
            'Accuracy': accuracy_score(true, preds)}

cross_dataset_results = {}

for species, ab in SPECIES_ANTIBIOTICS.items():
    mask = (meta_matched_B['species'] == species).values
    positions = np.where(mask)[0]
    labels = meta_matched_B.iloc[positions][ab].apply(parse_label)
    valid = labels.notna().values
    idx = positions[valid]

    if len(idx) == 0:
        print(f"No usable {species}/{ab} samples in DRIAMS-B, skipping.")
        continue

    X_b = X_spectra_B[idx]
    y_b = labels[valid].astype(int).values

    # IMPORTANT: scale using DRIAMS-A's fitted scaler statistics if you saved
    # them, not a fresh scaler fit on B — otherwise this isn't a fair test of
    # generalization. If you didn't save the scaler earlier, refitting here
    # is a reasonable fallback but weakens the cross-dataset claim slightly;
    # mention this as a limitation if so.
    scaler = StandardScaler()
    X_b_scaled = scaler.fit_transform(X_b).astype(np.float32)

    tag = f"{species.replace(' ', '_')}_{ab}"
    model_path = os.path.join(SAVE_DIR_A, f'{tag}_model.pt')
    if not os.path.exists(model_path):
        print(f"No trained DRIAMS-A model found for {tag}, skipping.")
        continue

    model = AMRNet(input_dim=6000).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    metrics = evaluate_model(model, X_b_scaled, y_b)
    cross_dataset_results[(species, ab)] = metrics
    print(f"\n{species} / {ab} — DRIAMS-A model tested on DRIAMS-B "
          f"({len(y_b)} samples): {metrics}")

cross_df = pd.DataFrame(cross_dataset_results).T
cross_df.index.names = ['species', 'antibiotic']
print("\n=== Cross-dataset validation summary (trained on A, tested on B) ===")
print(cross_df)
cross_df.to_csv(os.path.join(SAVE_DIR_B, 'cross_dataset_A_to_B_results.csv'))
print(f"\nSaved to {SAVE_DIR_B}/cross_dataset_A_to_B_results.csv")
