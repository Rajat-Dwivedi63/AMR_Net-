# =============================================================================
# TL-MALDI-AMR — ARO/CARD Ontology Interpretability (Colab)
#
# Run AFTER:
#   1) model training (needs the saved *_model.pt and *.npz files in
#      tl_maldi_amr_processed/ in your Drive), and
#   2) 3_card_aro_ontology_lookup.py (creates card_aro_lookup.csv, the
#      bin-to-gene table this script reads; +/-15 Da tolerance is set there).
# =============================================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SAVE_DIR = '/content/drive/MyDrive/tl_maldi_amr_processed'
TARGET_SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Staphylococcus aureus']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% Step 1 — Load the CARD/ARO lookup table built by 3_card_aro_ontology_lookup.py
LOOKUP_PATH = os.path.join(SAVE_DIR, 'card_aro_lookup.csv')
if not os.path.exists(LOOKUP_PATH):
    raise FileNotFoundError(
        f"{LOOKUP_PATH} not found. Run 3_card_aro_ontology_lookup.py first."
    )
card_aro_lookup = pd.read_csv(LOOKUP_PATH)
print(f"Loaded CARD/ARO lookup: {len(card_aro_lookup)} candidate bin matches")
for sp in TARGET_SPECIES:
    sp_matches = card_aro_lookup[card_aro_lookup['species'] == sp]
    print(f"  {sp}: {len(sp_matches)} matches, {sp_matches['gene'].nunique()} unique genes")



# %% Step 2 — Model architecture (must match training exactly)
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


# %% Step 3 — Load trained models and extract attention per sample
def get_top_attention_bins(model, x_sample, top_k=20):
    model.eval()
    x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, attn = model(x_tensor, return_attention=True)
    prob = torch.sigmoid(logits).item()
    attn = attn.cpu().numpy().flatten()
    top_idx = np.argsort(attn)[-top_k:][::-1]
    return top_idx, attn[top_idx], prob


SPECIES_ANTIBIOTICS = {
    'Escherichia coli': 'Ciprofloxacin',
    'Klebsiella pneumoniae': 'Ceftriaxone',
    'Staphylococcus aureus': 'Oxacillin',
}

all_explanations = []

for species, ab in SPECIES_ANTIBIOTICS.items():
    tag = f"{species.replace(' ', '_')}_{ab}"
    model_path = os.path.join(SAVE_DIR, f'{tag}_model.pt')
    npz_path = os.path.join(SAVE_DIR, f'{tag}.npz')

    if not (os.path.exists(model_path) and os.path.exists(npz_path)):
        print(f"Missing files for {tag}, skipping.")
        continue

    model = AMRNet(input_dim=6000).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    data = np.load(npz_path)

    X_test, y_test = data['X_test'], data['y_test']
    species_lookup = card_aro_lookup[card_aro_lookup['species'] == species]

    # %% Step 4 — Aggregate across ALL correctly-predicted Resistant test
    # samples, not just one — a single match could be coincidence.
    n_checked, n_with_match = 0, 0
    gene_hit_counts = {}

    for i in range(len(X_test)):
        if y_test[i] != 1:  # only look at true Resistant samples
            continue
        top_idx, top_weights, prob = get_top_attention_bins(model, X_test[i])
        if prob < 0.5:  # only look at samples the model correctly predicted R
            continue

        n_checked += 1
        matches = species_lookup[species_lookup['bin_index'].isin(top_idx)]
        if len(matches) > 0:
            n_with_match += 1
            for gene in matches['gene'].unique():
                gene_hit_counts[gene] = gene_hit_counts.get(gene, 0) + 1

    print(f"\n=== {species} / {ab} ===")
    print(f"Correctly-predicted Resistant test samples checked: {n_checked}")
    if n_checked > 0:
        print(f"Samples with at least one ARO/CARD attention match: "
              f"{n_with_match} ({100*n_with_match/n_checked:.1f}%)")
        if gene_hit_counts:
            print("Genes appearing in high-attention regions, by frequency:")
            for gene, count in sorted(gene_hit_counts.items(), key=lambda x: -x[1]):
                print(f"  {gene}: {count} samples ({100*count/n_checked:.1f}%)")
        else:
            print("No genes matched any high-attention bins for this pair.")

    all_explanations.append({
        'species': species, 'antibiotic': ab,
        'n_resistant_correct': n_checked,
        'n_with_ontology_match': n_with_match,
        'top_genes': dict(sorted(gene_hit_counts.items(), key=lambda x: -x[1])[:5]),
    })

explanations_df = pd.DataFrame(all_explanations)
explanations_df.to_csv(os.path.join(SAVE_DIR, 'ontology_interpretability_summary.csv'), index=False)
print(f"\nSaved summary to {SAVE_DIR}/ontology_interpretability_summary.csv")
