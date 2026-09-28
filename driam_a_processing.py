# =============================================================================
# TL-MALDI-AMR — Full Pipeline for COLAB 
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
print(f"Using device: {DEVICE}")

TARGET_SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Staphylococcus aureus']
SPECIES_ANTIBIOTICS = {
    'Escherichia coli': ['Ciprofloxacin'],
    'Klebsiella pneumoniae': ['Ceftriaxone'],
    'Staphylococcus aureus': ['Oxacillin'],
}
WORK_DIR = '/content/DRIAMS'
SAVE_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
os.makedirs(SAVE_DIR, exist_ok=True)


# %% Step 1 — Extract DRIAMS-A (disk operation, doesn't touch RAM much)
ZIP_PATH = '/content/drive/MyDrive/driams.zip'  # adjust to your actual path

if not os.path.exists(os.path.join(WORK_DIR, 'DRIAMS-A')):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(WORK_DIR)
    print("Extraction complete.")
else:
    print("Already extracted, skipping.")


# %% Step 2 — Load metadata (id folder), filter to target species
# Metadata CSVs are small — this step is not the RAM problem.
id_path = os.path.join(WORK_DIR, 'DRIAMS-A', 'id')
id_files = glob.glob(os.path.join(id_path, '**', '*.csv'), recursive=True)
print(f"Found {len(id_files)} metadata files")

# Sanity check: DRIAMS id folders sometimes contain BOTH an original and a
# "_clean" version of the same year (e.g. 2018.csv AND 2018_clean.csv).
# Print the list so you can visually confirm before we dedupe.
for f in sorted(id_files):
    print(f"  {os.path.basename(f)}")

meta = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in id_files], ignore_index=True
)
print(f"Raw concatenated metadata: {len(meta)} rows")

# Deduplicate by 'code' — if the same spectrum code appears more than once
# (e.g. from overlapping original + _clean files), keep only one row per
# code. This is what fixes the exact-2x row-count mismatch seen earlier.
n_before = len(meta)

# IMPORTANT: DRIAMS-A metadata genuinely has multiple rows per spectrum
# code — samples can be tested against different antibiotic panels, with
# different antibiotics filled in on different rows for the SAME code.
# A naive drop_duplicates(keep='first') would silently discard real R/S
# results that only appear on a later row for that code. Instead, we
# COALESCE: for each code, merge all its rows by taking, per column, the
# first value that isn't '-' (missing). Non-antibiotic columns (species
# etc.) are identical across a code's rows, so "first" is correct for them.
def coalesce_value(series):
    non_missing = series[series.astype(str) != '-']
    return non_missing.iloc[0] if len(non_missing) > 0 else series.iloc[0]

other_cols = [c for c in meta.columns if c != 'code']
meta = meta.groupby('code', as_index=False).agg(
    {c: coalesce_value for c in other_cols}
)
n_after = len(meta)
print(f"After coalescing duplicate rows by code: {n_after} unique codes "
      f"(was {n_before} raw rows) — merged, not dropped, so no R/S "
      f"labels were lost in the process")

meta_filtered = meta[meta['species'].isin(TARGET_SPECIES)].copy()
print(meta_filtered['species'].value_counts())

# Free the full (unfiltered) metadata — no longer needed
del meta
gc.collect()

# %% Step 3 — MEMORY-SAFE spectra loading
# Instead of pd.read_csv per file + growing a list, we:
#   1) Locate matching files first (cheap, just path strings)
#   2) Preallocate one float32 numpy array of the exact final size
#   3) Fill it in-place with lightweight numpy parsing
binned_path = os.path.join(WORK_DIR, 'DRIAMS-A', 'binned_6000')
binned_files = glob.glob(os.path.join(binned_path, '**', '*.txt'), recursive=True)
code_to_file = {os.path.splitext(os.path.basename(f))[0]: f for f in binned_files}
print(f"Found {len(binned_files)} binned spectrum files on disk")

# Only keep codes that actually have a matching spectrum file
matched_codes = [c for c in meta_filtered['code'] if c in code_to_file]
print(f"{len(matched_codes)} / {len(meta_filtered)} metadata rows have a matching spectrum")

N_BINS = 6000
n_samples = len(matched_codes)

# Preallocate — this is the key memory saving vs. building a DataFrame
# from a list of Series objects. float32 halves memory vs. float64.
X_spectra = np.empty((n_samples, N_BINS), dtype=np.float32)

# Your files are: "bin_index binned_intensity" (header row, whitespace-
# delimited, 2 columns). There's no actual m/z value in the file — just the
# bin position — so we reconstruct the real m/z axis from your project's
# confirmed binning scheme: 6000 bins, 3 Da width, spanning 2000-20000 Da.
mz_axis = 2000.0 + np.arange(N_BINS, dtype=np.float32) * 3.0

for i, code in enumerate(matched_codes):
    fpath = code_to_file[code]
    arr = np.loadtxt(fpath, skiprows=1, dtype=np.float32)  # cols: bin_index, binned_intensity
    X_spectra[i, :] = arr[:, 1]

    if i % 2000 == 0:
        print(f"  loaded {i}/{n_samples}")

print(f"Spectra matrix: {X_spectra.shape}, dtype={X_spectra.dtype}, "
      f"memory ~{X_spectra.nbytes / 1e9:.2f} GB")

# Save immediately to disk so this expensive step never needs re-running
np.save(os.path.join(SAVE_DIR, 'X_spectra.npy'), X_spectra)
np.save(os.path.join(SAVE_DIR, 'spectra_codes.npy'), np.array(matched_codes))

meta_matched = meta_filtered[meta_filtered['code'].isin(matched_codes)].copy()
meta_matched = meta_matched.set_index('code').loc[matched_codes].reset_index()
meta_matched.to_csv(os.path.join(SAVE_DIR, 'meta_matched.csv'), index=False)
np.save(os.path.join(SAVE_DIR, 'mz_axis.npy'), mz_axis)

print("Saved X_spectra.npy, meta_matched.csv, mz_axis.npy to Drive.")

# Free what we no longer need in this exact form
del meta_filtered, code_to_file, binned_files
gc.collect()


# %% Step 4 — Label parsing (handles composite DRIAMS labels like 'R(1), S(1)')
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


# %% Step 5 — Build per-species/antibiotic datasets ONE AT A TIME
# Reload from disk rather than trusting in-memory variables — if any cell
# above got re-run out of order (e.g. Step 2 re-run after Step 3), the
# in-memory meta_matched and X_spectra can silently drift out of sync in
# row count/order, causing exactly the IndexError seen above. Loading both
# fresh from the saved files guarantees they're the exact matched pair.
X_spectra = np.load(os.path.join(SAVE_DIR, 'X_spectra.npy'))
meta_matched = pd.read_csv(os.path.join(SAVE_DIR, 'meta_matched.csv'), low_memory=False)

# Robust alignment: match by actual code value, not just row count/order.
codes_path = os.path.join(SAVE_DIR, 'spectra_codes.npy')
if os.path.exists(codes_path):
    spectra_codes = np.load(codes_path, allow_pickle=True)
    code_to_row = {c: i for i, c in enumerate(spectra_codes)}

    # Coalesce meta_matched by code too, in case it still holds an old
    # duplicated version from before the Step 2 fix.
    other_cols = [c for c in meta_matched.columns if c != 'code']
    def coalesce_value(series):
        non_missing = series[series.astype(str) != '-']
        return non_missing.iloc[0] if len(non_missing) > 0 else series.iloc[0]
    meta_matched = meta_matched.groupby('code', as_index=False).agg(
        {c: coalesce_value for c in other_cols}
    )

    # Keep only codes present in both, in a consistent shared order
    common_codes = [c for c in meta_matched['code'] if c in code_to_row]
    meta_matched = meta_matched.set_index('code').loc[common_codes].reset_index()
    row_order = [code_to_row[c] for c in common_codes]
    X_spectra = X_spectra[row_order]

    print(f"Aligned by code: {len(meta_matched)} matched samples "
          f"(spectra_codes.npy found and used for exact alignment)")
else:
    print("WARNING: spectra_codes.npy not found (from a run before this fix) — "
          "falling back to row-count check only, which is less reliable.")
    if len(meta_matched) != X_spectra.shape[0]:
        meta_matched = meta_matched.drop_duplicates(subset='code', keep='first').reset_index(drop=True)

assert len(meta_matched) == X_spectra.shape[0], (
    f"STILL MISMATCHED: meta_matched has {len(meta_matched)} rows but "
    f"X_spectra has {X_spectra.shape[0]} rows. Delete meta_matched.csv, "
    f"X_spectra.npy, and spectra_codes.npy from {SAVE_DIR} and re-run "
    f"Step 1-3 fully from a fresh Runtime > Restart session."
)
print(f"Verified aligned: {len(meta_matched)} metadata rows == "
      f"{X_spectra.shape[0]} spectra rows")


def build_and_save_dataset(species, antibiotic, X_spectra, meta_matched, save_dir):
    # Reset index defensively — a CSV round-trip or upstream filtering can
    # leave a non-contiguous index, which would break positional alignment
    # between meta_matched and X_spectra below.
    meta_matched = meta_matched.reset_index(drop=True)

    mask = (meta_matched['species'] == species).values
    species_positions = np.where(mask)[0]

    labels = meta_matched.loc[species_positions, antibiotic].apply(parse_label)
    valid_mask = labels.notna().values
    idx = species_positions[valid_mask]

    X = X_spectra[idx]
    y = labels[valid_mask].astype(int).values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    tag = f"{species.replace(' ', '_')}_{antibiotic}"
    np.savez(os.path.join(save_dir, f'{tag}.npz'),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    print(f"{species} / {antibiotic}: train={len(y_train)}, val={len(y_val)}, "
          f"test={len(y_test)} -> saved {tag}.npz")

    del X, X_train, X_val, X_test
    gc.collect()

for species, antibiotics in SPECIES_ANTIBIOTICS.items():
    for ab in antibiotics:
        build_and_save_dataset(species, ab, X_spectra, meta_matched, SAVE_DIR)

# Now free the big spectra matrix — everything needed is saved as .npz
del X_spectra
gc.collect()
print("\nFreed X_spectra from memory. All per-pair datasets saved as .npz files.")
print("If Colab crashes from here on, re-run only from Step 6 — Steps 1-5 "
      "don't need to repeat since their outputs are saved to Drive.")


# %% Step 6 — Model architecture
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


# %% Step 7 — Train + evaluate, loading ONE .npz at a time
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
                print(f"  early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | val_AUROC {val_auroc:.4f}")

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
    return {'AUROC': roc_auc_score(true, probs), 'F1': f1_score(true, preds),
            'Accuracy': accuracy_score(true, preds)}

results = {}
for species, antibiotics in SPECIES_ANTIBIOTICS.items():
    for ab in antibiotics:
        tag = f"{species.replace(' ', '_')}_{ab}"
        npz_path = os.path.join(SAVE_DIR, f'{tag}.npz')
        if not os.path.exists(npz_path):
            print(f"Missing {npz_path}, skipping — re-run Step 5 for this pair.")
            continue

        data = np.load(npz_path)
        print(f"\nTraining: {species} / {ab}")
        model, _ = train_model(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
        metrics = evaluate_model(model, data['X_test'], data['y_test'])
        results[(species, ab)] = metrics
        print(f"  -> {metrics}")

        # Save model weights, then free everything for this pair
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{tag}_model.pt'))
        del model, data
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

results_df = pd.DataFrame(results).T
results_df.index.names = ['species', 'antibiotic']
print("\n=== Summary ===")
print(results_df)
results_df.to_csv(os.path.join(SAVE_DIR, 'own_pipeline_results.csv'))
print(f"\nAll results and model weights saved to {SAVE_DIR}")
