# =============================================================================
# build_card_aro_lookup.py
# Constructs the ARO/CARD ontology lookup table needed for step 10 of the
# main pipeline (tl_maldi_amr_pipeline.py).
#
# WHAT THIS DOES:
#   CARD contains gene/protein sequences linked to ARO ontology terms, but
#   NO mass spectrometry data. To connect a resistance gene to a specific
#   bin in your binned_6000 MALDI-TOF spectrum, we:
#     1) Filter CARD's protein sequences to your 3 target species
#     2) Compute each protein's theoretical average mass from its sequence
#     3) Match that mass to the nearest bin in your DRIAMS-A m/z axis
#     4) Attach the ARO term / drug class / resistance mechanism to that bin
#
# HONEST LIMITATION — read this before trusting the output:
#   - This is a theoretical-mass proxy, not a validated biomarker mapping.
#   - Real MALDI-TOF peaks rarely match theoretical mass exactly (PTMs,
#     truncation, charge state, calibration drift commonly shift peaks by
#     tens to low-hundreds of Da).
#   - Standard bacterial biotyping MALDI-TOF range is roughly 2,000-20,000 Da
#     and mostly detects abundant ribosomal/housekeeping proteins — most
#     resistance proteins (many are >20 kDa, e.g. beta-lactamases ~25-30 kDa)
#     will NOT appear in this range at all and will correctly get no match.
#   - Tolerance width (MZ_TOLERANCE_DA below) directly trades off precision
#     vs. recall of matches — treat matches as hypotheses to validate against
#     literature-reported AMR biomarker peaks, not confirmed identifications.
#   - Multiple proteins can plausibly fall in the same bin/tolerance window;
#     the output includes ALL candidates per bin, not a forced single answer.
#
# This is very much a research-design decision worth discussing with
# Dr. Jain before treating it as the paper's core interpretability claim.
# =============================================================================

import os
import re
import glob
import pandas as pd
from google.colab import drive

# Mount Drive (always this exact call)
drive.mount('/content/drive')

# Auto-locate the CARD files anywhere in your Drive, so no path needs editing.
# Needs: protein_fasta_protein_homolog_model.fasta and aro_index.tsv
def find_file(root, name):
    for dirpath, _, filenames in os.walk(root):
        if name in filenames:
            return os.path.join(dirpath, name)
    return None

FASTA_PATH = find_file('/content/drive/MyDrive', 'protein_fasta_protein_homolog_model.fasta')
ARO_INDEX_PATH = find_file('/content/drive/MyDrive', 'aro_index.tsv')
if FASTA_PATH is None or ARO_INDEX_PATH is None:
    raise FileNotFoundError(
        "Could not find protein_fasta_protein_homolog_model.fasta and/or "
        "aro_index.tsv in your Drive. Upload them from the main CARD download "
        "(card.mcmaster.ca/download) and re-run."
    )
print(f"Using FASTA: {FASTA_PATH}\nUsing ARO index: {ARO_INDEX_PATH}")

OUTPUT_PATH = '/content/drive/MyDrive/tl_maldi_amr_processed/card_aro_lookup.csv'

TARGET_SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Staphylococcus aureus']

MZ_RANGE_MIN, MZ_RANGE_MAX = 2000, 20000
MZ_TOLERANCE_DA = 15  # matches the value reported in the paper (+/-15 Da)


# %% Step 1 — Average amino acid residue masses (Da), for theoretical mass calc
AA_MASS = {
    'G': 57.0519, 'A': 71.0788, 'S': 87.0782, 'P': 97.1167, 'V': 99.1326,
    'T': 101.1051, 'C': 103.1388, 'L': 113.1594, 'I': 113.1594, 'N': 114.1038,
    'D': 115.0886, 'Q': 128.1307, 'K': 128.1741, 'E': 129.1155, 'M': 131.1926,
    'H': 137.1411, 'F': 147.1766, 'R': 156.1875, 'Y': 163.1760, 'W': 186.2132,
}
WATER_MASS = 18.0153

def theoretical_mass(seq):
    seq = seq.upper().strip()
    total = sum(AA_MASS.get(aa, 0.0) for aa in seq)
    return total + WATER_MASS


# %% Step 2 — Parse CARD protein FASTA, filter to target species
def parse_card_fasta(fasta_path, target_species):
    """
    CARD header format:
    >gb|ACCESSION|ARO:XXXXXXX|Gene_Name [Species Name]
    """
    records = []
    header = None
    seq_lines = []

    def flush():
        if header is None:
            return
        m = re.match(r'>gb\|([^|]+)\|(ARO:\d+)\|(.+?)\s*\[(.+?)\]', header)
        if not m:
            return
        accession, aro_acc, gene_name, species = m.groups()
        if species not in target_species:
            return
        seq = ''.join(seq_lines)
        records.append({
            'protein_accession': accession,
            'aro_accession': aro_acc,
            'gene_name': gene_name,
            'species': species,
            'sequence_length': len(seq),
            'theoretical_mass_da': theoretical_mass(seq),
        })

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush()
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        flush()  # last record

    return pd.DataFrame(records)

fasta_path = FASTA_PATH
card_proteins = parse_card_fasta(fasta_path, TARGET_SPECIES)
print(f"Parsed {len(card_proteins)} CARD protein records for target species")
print(card_proteins['species'].value_counts())


# %% Step 3 — Merge in ARO ontology metadata (drug class, mechanism, gene family)
aro_index = pd.read_csv(ARO_INDEX_PATH, sep='\t')

aro_meta = aro_index[[
    'ARO Accession', 'ARO Name', 'AMR Gene Family',
    'Drug Class', 'Resistance Mechanism', 'CARD Short Name'
]].drop_duplicates()

card_proteins = card_proteins.merge(
    aro_meta, left_on='aro_accession', right_on='ARO Accession', how='left'
)
print(f"After ARO metadata merge: {card_proteins.shape}")
print(card_proteins[['gene_name', 'species', 'theoretical_mass_da', 'Drug Class']].head(10))


# %% Step 4 — Filter to the MALDI-TOF measurable mass range
# Bruker Biotyper linear-mode range is typically ~2,000-20,000 Da.
# Confirm this against your actual DRIAMS-A binned_6000 m/z axis in step 5.
MZ_RANGE_MIN, MZ_RANGE_MAX = 2000, 20000

in_range = card_proteins[
    (card_proteins['theoretical_mass_da'] >= MZ_RANGE_MIN) &
    (card_proteins['theoretical_mass_da'] <= MZ_RANGE_MAX)
].copy()

print(f"\n{len(in_range)} / {len(card_proteins)} CARD proteins fall in the "
      f"MALDI-TOF measurable range ({MZ_RANGE_MIN}-{MZ_RANGE_MAX} Da).")
print("This is expected to be a small fraction — most resistance proteins "
      "(e.g. many beta-lactamases) are heavier than the biotyping range.")


# %% Step 5 — Reconstruct the m/z axis
# CONFIRMED: this dataset's binned_6000 files contain only
# "bin_index  binned_intensity" (no actual m/z value stored). The real m/z
# axis must be reconstructed from your project's confirmed binning scheme:
# 6000 bins, 3 Da width, spanning 2000-20000 Da. This matches the file
# structure verified directly against your uploaded data.
import numpy as np
mz_axis = MZ_RANGE_MIN + np.arange(6000, dtype=np.float32) * 3.0
print(f"\nReconstructed m/z axis: {mz_axis.min():.1f} - {mz_axis.max():.1f} Da, "
      f"{len(mz_axis)} bins (3 Da width, matches confirmed binning scheme)")


# %% Step 6 — Match each CARD protein's theoretical mass to the nearest bin(s)
import numpy as np

lookup_rows = []
for _, row in in_range.iterrows():
    target_mass = row['theoretical_mass_da']
    diffs = np.abs(mz_axis - target_mass)
    within_tol = np.where(diffs <= MZ_TOLERANCE_DA)[0]

    for bin_idx in within_tol:
        lookup_rows.append({
            'bin_index': int(bin_idx),
            'mz_value': float(mz_axis[bin_idx]),
            'mass_diff_da': float(diffs[bin_idx]),
            'gene': row['gene_name'],
            'species': row['species'],
            'aro_accession': row['aro_accession'],
            'aro_term': row['ARO Name'],
            'gene_family': row['AMR Gene Family'],
            'drug_class': row['Drug Class'],
            'resistance_mechanism': row['Resistance Mechanism'],
            'card_short_name': row['CARD Short Name'],
        })

card_aro_lookup = pd.DataFrame(lookup_rows).sort_values(['bin_index', 'mass_diff_da'])
print(f"\nBuilt lookup table: {len(card_aro_lookup)} bin-to-ARO candidate matches, "
      f"covering {card_aro_lookup['bin_index'].nunique()} unique bins")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
card_aro_lookup.to_csv(OUTPUT_PATH, index=False)
print(f"Saved to {OUTPUT_PATH}")
print("\nThis is now ready to be loaded by step 10 of tl_maldi_amr_pipeline.py "
      "(CARD_LOOKUP_PATH).")

# %% Step 7 — Sanity-check the results before trusting them
print("\n=== Sample matches per species ===")
for sp in TARGET_SPECIES:
    sp_matches = card_aro_lookup[card_aro_lookup['species'] == sp]
    print(f"\n{sp}: {len(sp_matches)} candidate matches, "
          f"{sp_matches['gene'].nunique()} unique genes")
    print(sp_matches[['gene', 'mz_value', 'mass_diff_da', 'drug_class']].head(5))

print("\n=== Next step ===")
print("Cross-check a handful of these matches against published MALDI-TOF AMR "
      "biomarker literature for these species (e.g. known mecA/PBP2a-related "
      "peaks for S. aureus oxacillin resistance) before using this as your "
      "paper's interpretability mechanism. If matches don't hold up, narrowing "
      "MZ_TOLERANCE_DA or restricting to specific gene families is the next lever.")
