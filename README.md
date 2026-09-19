# TL-MALDI-AMR

Deep learning pipeline for antimicrobial resistance (AMR) prediction from MALDI-TOF mass spectrometry data, with ARO/CARD ontology-based interpretability in place of statistical feature-importance methods.

## Overview

Antimicrobial resistance is a growing global health threat, and rapid resistance profiling can meaningfully improve treatment decisions. MALDI-TOF mass spectrometry is already used routinely for bacterial species identification — this project investigates whether the same spectra can also predict antibiotic resistance, and whether model predictions can be explained through a curated biological ontology (CARD/ARO) rather than abstract statistical importance scores.

This is an active research project developed as part of a research internship at NIT Kurukshetra, under the supervision of Dr. Sarika Jain.

## Scope

Three clinically relevant species/antibiotic pairs:

| Species | Antibiotic |
|---|---|
| *Escherichia coli* | Ciprofloxacin |
| *Klebsiella pneumoniae* | Ceftriaxone |
| *Staphylococcus aureus* | Oxacillin |

## Datasets

| Dataset | Role | Status |
|---|---|---|
| [DRIAMS-A](https://www.dora.lib4ri.ch/eawag/islandora/object/eawag:19773) | Primary training/evaluation | Fully processed (18,235 unique isolates) |
| DRIAMS-B | Cross-dataset validation | Fully processed (5,897 isolates, 718 spectra-matched) |
| [PATRIC / BV-BRC](https://www.bv-brc.org/) | Independent phenotype-rate consistency check | 75,000 phenotype records compared against DRIAMS-A |
| [CARD](https://card.mcmaster.ca/) | Ontology reference for interpretability | Fully integrated (1,297 relevant protein records) |

Raw dataset files are not included in this repository due to size and licensing — see [Data Access](#data-access) below.

## Model Architecture

**AMRNet** — a deep neural network with an input-level attention layer over the 6,000-bin MALDI-TOF spectrum (3 Da bin width, 2,000–20,000 Da range), followed by a 3-layer backbone (1024 → 256 → 64) with batch normalization, ReLU, and dropout. The attention layer serves a dual purpose: improving classification and providing the mechanism used for ontology-based interpretability.

- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=1e-3, weight decay=1e-5)
- **Early stopping:** patience=15 on validation AUROC

## Results

### Classification (DRIAMS-A test set)

| Species / Antibiotic | AUROC | F1 | Accuracy |
|---|---|---|---|
| E. coli / Ciprofloxacin | 0.828 | 0.663 | 0.831 |
| K. pneumoniae / Ceftriaxone | 0.803 | 0.561 | 0.890 |
| S. aureus / Oxacillin | 0.885 | 0.627 | 0.889 |

### Cross-Dataset Validation (trained on DRIAMS-A, tested on DRIAMS-B)

| Species / Antibiotic | AUROC | Notes |
|---|---|---|
| E. coli / Ciprofloxacin | 0.827 | Strong generalization — nearly identical to DRIAMS-A |
| S. aureus / Oxacillin | 0.700 | Moderate drop, consistent with known cross-site effects |
| K. pneumoniae / Ceftriaxone | 0.421 | Inconclusive — only 17 resistant samples in test set |

## Ontology-Based Interpretability

In place of SHAP-style statistical feature importance, model attention weights are mapped to CARD/ARO resistance genes via theoretical protein mass matching against the MALDI-TOF measurable range. Only 34 of 1,297 relevant CARD resistance proteins fall within the detectable mass range (2,000–20,000 Da) — a physical limitation of the technique itself, not a shortcoming of the method.

| Species / Antibiotic | Samples Checked | Ontology Match | Top Gene |
|---|---|---|---|
| E. coli / Ciprofloxacin | 122 | 70 (57.4%) | dfrA9 (36.9%) |
| K. pneumoniae / Ceftriaxone | 30 | 0 (0.0%) | none |
| S. aureus / Oxacillin | 53 | 7 (13.2%) | qacJ (5.7%) |

**Important caveat:** these ontology matches identify biologically plausible candidate genes based on theoretical mass, not confirmed causal relationships. For example, `dfrA9` does not directly cause ciprofloxacin resistance — it confers trimethoprim resistance — but `dfrA`-family genes are documented as common passengers on multidrug-resistance plasmids in *E. coli*, so this may reflect a genuine indirect co-resistance marker. This requires genomic confirmation to verify directly and should currently be read as a research hypothesis, not a validated finding.

## Repository Structure

```
TL-MALDI-AMR/
├── data_processing/        # DRIAMS-A / DRIAMS-B extraction & preprocessing scripts
├── ontology/                # CARD/ARO parsing and bin-to-gene mapping scripts
├── models/                  # AMRNet architecture, training, evaluation
├── interpretability/        # Attention-to-ontology mapping and analysis
├── results/                 # Saved metrics, trained model weights, output CSVs
├── notebooks/                # Exploratory / Colab notebooks
└── README.md
```


## Data Access

- DRIAMS-A / DRIAMS-B: available from the [DRIAMS repository](https://www.dora.lib4ri.ch/eawag/islandora/object/eawag:19773) (institutional access required)
- CARD: available from [card.mcmaster.ca](https://card.mcmaster.ca/download) (free for non-commercial/research use — see CARD's own license terms)
- PATRIC/BV-BRC: available via the [BV-BRC API](https://www.bv-brc.org/)

## Current Status

- [x] SOTA baseline reproduction (MSDeepAMR, Weis et al.)
- [x] AMRNet trained and evaluated on DRIAMS-A
- [x] Cross-dataset validation on DRIAMS-B
- [x] ARO/CARD ontology interpretability pipeline
- [x] Independent phenotype-rate check against PATRIC
- [ ] Genomic confirmation of candidate co-resistance markers
- [ ] Class imbalance mitigation for K. pneumoniae
- [ ] Manuscript preparation

## Author

**Rajat Dwivedi** — B.Tech Information Technology, Global Institute of Technology, Jaipur
Research Internship under **Dr. Sarika Jain**, NIT Kurukshetra

## License
MIT license
