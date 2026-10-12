# =============================================================================
# card_mass_tolerance_sweep.py
# Section V-E: Mass-Tolerance Sensitivity Analysis (±1, ±3, ±5, ±15 Da)
# Uses pre-computed mass_diff_da from card_aro_lookup.csv
# =============================================================================

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
LOOKUP_CSV = os.path.join(DATA_DIR, 'card_aro_lookup.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'card_tolerance_sweep_results.csv')

PAIRS = [
    ('Escherichia_coli', 'Ciprofloxacin'),
    ('Klebsiella_pneumoniae', 'Ceftriaxone'),
    ('Staphylococcus_aureus', 'Oxacillin')
]

# Standard 6,000-bin linear mass axis: 2,000 to 20,000 Da (3 Da bin width)
MZ_AXIS = 2000.0 + np.arange(6000, dtype=np.float32) * 3.0


# --- 1. AMRNet Architecture ---
class AMRNet(nn.Module):
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
        return self.classifier(self.backbone(x * attn_weights)).squeeze(-1)


# --- 2. Load CARD Reference Lookup ---
card_df = pd.read_csv(LOOKUP_CSV)
print(f"Loaded CARD lookup table with {len(card_df)} reference mappings.")
print(f"Columns present: {list(card_df.columns)}")


# --- 3. Extract Top 1% Salient Attention Bins from Resistant Isolates ---
salient_bins_by_cohort = {}

print("\nExtracting salient attention peaks from resistant test isolates...")
for sp, ab in PAIRS:
    tag = f"{sp}_{ab}"
    npz_file = os.path.join(DATA_DIR, f"{tag}.npz")
    model_file = os.path.join(DATA_DIR, f"{tag}_model.pt")

    if not (os.path.exists(npz_file) and os.path.exists(model_file)):
        print(f"Skipping {tag}: model or data missing.")
        continue

    data = np.load(npz_file)
    X_te, y_te = data['X_test'], data['y_test']
    
    # Restrict to resistant isolates
    res_mask = (y_te == 1)
    if res_mask.sum() == 0:
        continue
    X_res = X_te[res_mask]

    model = AMRNet(input_dim=X_res.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(X_res, dtype=torch.float32).to(DEVICE)
        attn = model.attention(xb).cpu().numpy()

    # Mean attention profile across resistant cohort
    mean_attn = attn.mean(axis=0)

    # Top 1% salient bins (top 60 bins out of 6,000)
    top_thresh = np.percentile(mean_attn, 99.0)
    top_bins = np.where(mean_attn >= top_thresh)[0]
    salient_bins_by_cohort[tag] = set(top_bins)
    print(f"  [{sp} / {ab}] Extracted {len(top_bins)} top-attention bins (threshold = {top_thresh:.4f})")


# --- 4. Mass-Tolerance Sensitivity Sweep (±1, ±3, ±5, ±15 Da) ---
TOLERANCES = [1.0, 3.0, 5.0, 15.0]
sweep_rows = []

for tol in TOLERANCES:
    # Filter global reference table by tolerance
    tol_card = card_df[card_df['mass_diff_da'] <= tol]
    
    total_unique_genes = tol_card['gene'].nunique()
    total_bins_matched = tol_card['bin_index'].nunique()
    
    cohort_gene_counts = {}
    qacj_found = False
    dfra9_found = False

    for sp, ab in PAIRS:
        tag = f"{sp}_{ab}"
        salient_set = salient_bins_by_cohort.get(tag, set())

        # Intersect salient bins with tolerance-filtered CARD matches
        cohort_matches = tol_card[tol_card['bin_index'].isin(salient_set)]
        cohort_gene_counts[tag] = cohort_matches['gene'].nunique()

        # Check for specific biological markers
        matched_gene_names = cohort_matches['gene'].astype(str).tolist()
        if 'Staphylococcus' in tag:
            if any('qacj' in g.lower() for g in matched_gene_names):
                qacj_found = True
        if 'Escherichia' in tag:
            if any('dfra9' in g.lower() for g in matched_gene_names):
                dfra9_found = True

    sweep_rows.append({
        'Tolerance Window': f"± {tol:.0f} Da",
        'Total CARD Matches': len(tol_card),
        'Unique Genes (Window)': total_unique_genes,
        'Bins Matched (Global)': total_bins_matched,
        'S. aureus Salient Genes': cohort_gene_counts.get('Staphylococcus_aureus_Oxacillin', 0),
        'E. coli Salient Genes': cohort_gene_counts.get('Escherichia_coli_Ciprofloxacin', 0),
        'K. pneumoniae Salient Genes': cohort_gene_counts.get('Klebsiella_pneumoniae_Ceftriaxone', 0),
        'qacJ Retained': 'Yes' if qacj_found else 'No',
        'dfrA9 Retained': 'Yes' if dfra9_found else 'No'
    })

# --- 5. Summary Table Output ---
df_sweep = pd.DataFrame(sweep_rows)
df_sweep.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 80)
print("TABLE IV: MASS-TOLERANCE SENSITIVITY SWEEP (ARO / CARD GROUNDING)")
print("=" * 80)
print(df_sweep.to_string(index=False))
print("=" * 80)
print(f"Results saved to: {OUTPUT_CSV}\n")
