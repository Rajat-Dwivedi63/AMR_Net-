# =============================================================================
# AMRNet: Multi-Center External Validation (DRIAMS-B)
#
# Pipeline Overview:
#   Step 1: Extract DRIAMS-B spectral archive.
#   Step 2: Metadata parsing and label coalescing.
#   Step 3: Memory-mapped float32 spectral feature extraction.
#   Step 4: Phenotypic resistance label parsing.
#   Step 5: AMRNet model architecture definition.
#   Step 6: Cross-dataset evaluation (models trained on DRIAMS-A tested on DRIAMS-B).
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

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

WORK_DIR = '/content/DRIAMS_B'
SAVE_DIR_A = '/content/drive/MyDrive/tl_maldi_amr_processed'
SAVE_DIR_B = '/content/drive/MyDrive/tl_maldi_amr_processed_driamsB'
ZIP_PATH_B = '/content/drive/MyDrive/TL-MALDI-AMR/data/DRIAMS-B.zip'

os.makedirs(SAVE_DIR_B, exist_ok=True)


# =============================================================================
# Step 1: Extract DRIAMS-B Spectral Archive
# =============================================================================
if not os.path.exists(ZIP_PATH_B):
    raise FileNotFoundError(f"Archive not found: {ZIP_PATH_B}")

if not os.path.exists(os.path.join(WORK_DIR, 'id')) and \
   not os.path.exists(os.path.join(WORK_DIR, 'DRIAMS-B', 'id')):
    with zipfile.ZipFile(ZIP_PATH_B, 'r') as zf:
        zf.extractall(WORK_DIR)
    print("DRIAMS-B extraction complete.")
else:
    print("Dataset already extracted, skipping.")

# Resolve nested directory structure if present
if os.path.exists(os.path.join(WORK_DIR, 'DRIAMS-B', 'id')):
    DRIAMS_B_ROOT = os.path.join(WORK_DIR, 'DRIAMS-B')
elif os.path.exists(os.path.join(WORK_DIR, 'id')):
    DRIAMS_B_ROOT = WORK_DIR
else:
    raise FileNotFoundError(f"Could not locate metadata directory under {WORK_DIR}")

print(f"Using DRIAMS-B root: {DRIAMS_B_ROOT}")


# =============================================================================
# Step 2: Metadata Parsing and Coalescing
# =============================================================================
id_path = os.path.join(DRIAMS_B_ROOT, 'id')
id_files = glob.glob(os.path.join(id_path, '**', '*.csv'), recursive=True)
print(f"Found {len(id_files)} DRIAMS-B metadata files")

meta = pd.concat([pd.read_csv(f, low_memory=False) for f in id_files], ignore_index=True)

def coalesce_value(series):
    non_missing = series[series.astype(str) != '-']
    return non_missing.iloc[0] if len(non_missing) > 0 else series.iloc[0]

other_cols = [c for c in meta.columns if c != 'code']
meta = meta.groupby('code', as_index=False).agg({c: coalesce_value for c in other_cols})

meta_filtered = meta[meta['species'].isin(TARGET_SPECIES)].copy()
print(meta_filtered['species'].value_counts())

del meta
gc.collect()


# =============================================================================
# Step 3: Memory-Mapped Spectral Extraction (float32)
# =============================================================================
binned_path = os.path.join(DRIAMS_B_ROOT, 'binned_6000')
binned_files = glob.glob(os.path.join(binned_path, '**', '*.txt'), recursive=True)
code_to_file = {os.path.splitext(os.path.basename(f))[0]: f for f in binned_files}

matched_codes = [c for c in meta_filtered['code'] if c in code_to_file]
print(f"{len(matched_codes)} / {len(meta_filtered)} metadata records matched with spectral files.")

N_BINS = 6000
n_samples = len(matched_codes)
X_spectra_B = np.empty((n_samples, N_BINS), dtype=np.float32)

for i, code in enumerate(matched_codes):
    fpath = code_to_file[code]
    arr = np.loadtxt(fpath, skiprows=1, dtype=np.float32)
    X_spectra_B[i, :] = arr[:, 1]
    if i % 2000 == 0:
        print(f"  Loaded {i}/{n_samples}")

np.save(os.path.join(SAVE_DIR_B, 'X_spectra.npy'), X_spectra_B)
np.save(os.path.join(SAVE_DIR_B, 'spectra_codes.npy'), np.array(matched_codes))

meta_matched_B = meta_filtered[meta_filtered['code'].isin(matched_codes)].copy()
meta_matched_B = meta_matched_B.reset_index(drop=True)
meta_matched_B.to_csv(os.path.join(SAVE_DIR_B, 'meta_matched.csv'), index=False)

del meta_filtered, code_to_file, binned_files
gc.collect()


# =============================================================================
# Step 4: Phenotypic Resistance Label Parsing
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


# =============================================================================
# Step 5: Dataset Loader and AMRNet Architecture
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
# Step 6: Cross-Dataset Validation (DRIAMS-A Models on DRIAMS-B)
# =============================================================================
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

cross_dataset_results = {}

for species, ab in SPECIES_ANTIBIOTICS.items():
    mask = (meta_matched_B['species'] == species).values
    positions = np.where(mask)[0]
    labels = meta_matched_B.iloc[positions][ab].apply(parse_label)
    valid = labels.notna().values
    idx = positions[valid]

    if len(idx) == 0:
        print(f"No valid {species}/{ab} records in DRIAMS-B, skipping.")
        continue

    X_b = X_spectra_B[idx]
    y_b = labels[valid].astype(int).values

    scaler = StandardScaler()
    X_b_scaled = scaler.fit_transform(X_b).astype(np.float32)

    tag = f"{species.replace(' ', '_')}_{ab}"
    model_path = os.path.join(SAVE_DIR_A, f'{tag}_model.pt')
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found for {tag}, skipping.")
        continue

    model = AMRNet(input_dim=6000).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    metrics = evaluate_model(model, X_b_scaled, y_b)
    cross_dataset_results[(species, ab)] = metrics
    print(f"\n{species} / {ab} — DRIAMS-B Evaluation ({len(y_b)} isolates): {metrics}")

cross_df = pd.DataFrame(cross_dataset_results).T
cross_df.index.names = ['species', 'antibiotic']
print("\nCross-Dataset Validation Summary:")
print(cross_df)
cross_df.to_csv(os.path.join(SAVE_DIR_B, 'cross_dataset_A_to_B_results.csv'))
print(f"Results saved to {SAVE_DIR_B}/cross_dataset_A_to_B_results.csv")
