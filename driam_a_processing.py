from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np

ROOT = "/content/drive/MyDrive/TL-MALDI-AMR"

targets = {
    "E. coli":       ("e_coli_driams_a_bin3_2000_20000Da.csv",      "Ciprofloxacin"),
    "K. pneumoniae": ("k_pneumoniae_driams_a_bin3_2000_20000Da.csv", "Ceftriaxone"),
    "S. aureus":     ("s_aureus_driams_a_bin3_2000_20000Da.csv",     "Oxacillin"),
}

save_names = {
    "E. coli":       "ecoli_cipro",
    "K. pneumoniae": "kpneumo_ceftriaxone",
    "S. aureus":     "saureus_oxacillin",
}

print("Ready")



print("REAL STATISTICS FOR DATASET TABLE")
print("=" * 50)

for species, (filename, drug) in targets.items():
    path = f"{ROOT}/data/DRIAMS-A/id/{filename}"
    df   = pd.read_csv(path)

    bin_cols   = [c for c in df.columns if str(c).replace('.','').isnumeric()]
    label_cols = [c for c in df.columns if c not in bin_cols]

    print(f"\n{species}")
    print(f"  Total isolates     : {len(df)}")
    print(f"  Spectral features  : {len(bin_cols)}")
    print(f"  Antibiotics tested : {label_cols}")

    for drug_col in label_cols:
        counts = df[drug_col].value_counts()
        r = int(counts.get(1.0, 0))
        s = int(counts.get(0.0, 0))
        if r + s > 0:
            print(f"  {drug_col:25s}  R={r:5d}  S={s:5d}  total={r+s}")


os.makedirs(f"{ROOT}/data/DRIAMS-A/processed", exist_ok=True)

for species, (filename, drug) in targets.items():
    path    = f"{ROOT}/data/DRIAMS-A/id/{filename}"
    df      = pd.read_csv(path)
    bin_cols = [c for c in df.columns if str(c).replace('.','').isnumeric()]
    df_drug  = df[df[drug].isin([0.0, 1.0])].copy()

    X = df_drug[bin_cols].values.astype(float)
    y = df_drug[drug].values.astype(int)

    name = save_names[species]
    np.save(f"{ROOT}/data/DRIAMS-A/processed/X_{name}.npy", X)
    np.save(f"{ROOT}/data/DRIAMS-A/processed/y_{name}.npy", y)

    print(f"\n{species} / {drug}")
    print(f"  Samples    : {len(y)}")
    print(f"  Resistant  : {y.sum()}")
    print(f"  Susceptible: {(y==0).sum()}")
    print(f"  Saved as   : X_{name}.npy  y_{name}.npy")
