# TL-MALDI-AMR

Deep learning pipeline for antimicrobial resistance (AMR) prediction from MALDI-TOF mass spectrometry data, with ARO/CARD ontology-based interpretability and a genome-level co-occurrence check of the resulting hypotheses.

## Overview

Antimicrobial resistance is a growing global health threat, and rapid resistance profiling can meaningfully improve treatment decisions. MALDI-TOF mass spectrometry is already used routinely for bacterial species identification. This project investigates whether the same spectra can also predict antibiotic resistance, and whether model predictions can be explained through a curated biological ontology (CARD/ARO) rather than abstract statistical importance scores.

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
| [CARD](https://card.mcmaster.ca/) / ARO | Ontology reference for interpretability | Integrated (1,297 relevant protein records, 34 in measurable range) |
| CARD Genome Collection | Genomic co-occurrence validation | Analyzed (250,685 sequenced genomes) |

Raw dataset files are not included in this repository due to size and licensing. See [Data Access](#data-access).

## Model Architecture

**AMRNet** is a deep neural network with an input-level attention layer over the 6,000-bin MALDI-TOF spectrum (3 Da bin width, 2,000-20,000 Da range), followed by a 3-layer backbone (1024 → 256 → 64) with batch normalization, ReLU, and dropout. The attention layer serves a dual purpose: improving classification and providing the mechanism used for ontology-based interpretability.

- **Loss:** BCEWithLogitsLoss
- **Optimizer:** Adam (lr=1e-3, weight decay=1e-5)
- **Class imbalance:** WeightedRandomSampler
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
| E. coli / Ciprofloxacin | 0.827 | Strong generalization, nearly identical to DRIAMS-A |
| S. aureus / Oxacillin | 0.700 | Moderate drop, consistent with known cross-site effects |
| K. pneumoniae / Ceftriaxone | 0.421 | Inconclusive: only 17 resistant samples in the external test set |

## Ontology-Based Interpretability

Model attention weights are mapped to CARD/ARO resistance genes via theoretical protein mass matching (±15 Da) against the MALDI-TOF measurable range. Only 34 of 1,297 relevant CARD resistance proteins fall within the detectable mass range (2,000-20,000 Da). This is a physical limit of the technique, not a shortcoming of the method.

| Species / Antibiotic | Samples Checked | Ontology Match | Top Gene |
|---|---|---|---|
| E. coli / Ciprofloxacin | 122 | 70 (57.4%) | dfrA9 (36.9%) |
| K. pneumoniae / Ceftriaxone | 30 | 0 (0.0%) | none |
| S. aureus / Oxacillin | 53 | 7 (13.2%) | qacJ (5.7%) |

## Genomic Validation

The two candidate genes were checked for co-occurrence with their biologically plausible partner genes across CARD's genome collection (250,685 sequenced genomes).

| Candidate gene | Carriers / Total genomes | Partner rate in carriers | Baseline rate |
|---|---|---|---|
| *qacJ* (S. aureus) | 590 / 42,735 | 64.7% carry *mecA* | 24.7% |
| *dfrA9* (E. coli) | 1 / 45,170 | 1/1 fluoroquinolone class | 86.1% |

- **qacJ:** supported. Carriers are about 2.6 times more likely to also carry *mecA* than a random *S. aureus* genome, consistent with co-localization on shared mobile genetic elements. *mecA* itself (PBP2a, ~76 kDa) is too heavy to detect directly, so *qacJ* is a plausible indirect marker.
- **dfrA9:** inconclusive. Only one carrier genome exists, and the baseline fluoroquinolone-resistance annotation rate among non-carriers is already 86.1%, leaving little room to detect an effect.
- **Limitation:** this is population-level evidence, not isolate-specific. DRIAMS contains spectra only, so no genome sequence exists for the exact isolates the model classified.

## Repository Structure

```
TL-MALDI-AMR/
├── data_processing/
│   ├── 1_driams_a_processing.py
│   └── 2_driams_b_processing_and_validation.py
├── ontology/
│   └── 3_card_aro_ontology_lookup.py
├── interpretability/
│   └── colab_ontology_interpretability.py
├── validation/
│   └── analyze_co_occurrence.py
├── paper/
│   ├── TL-MALDI-AMR.tex
│   └── TL-MALDI-AMR.pdf
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/Rajat-Dwivedi63/TL-MALDI-AMR.git
cd TL-MALDI-AMR
pip install -r requirements.txt
```

The scripts were developed in Google Colab and contain Google Drive paths that you will need to edit to match your own data locations.

## Data Access

- **DRIAMS-A / DRIAMS-B:** available from the [DRIAMS repository](https://www.dora.lib4ri.ch/eawag/islandora/object/eawag:19773) (see the repository for access terms).
- **CARD (ontology and protein FASTA):** [card.mcmaster.ca/download](https://card.mcmaster.ca/download), main data download.
- **CARD genome collection:** the separate "Prevalence, Resistomes & Variants" download on the same page (`card-genomes.txt.gz`).

CARD is free for non-commercial research use under its own license terms.

## Current Status

- [x] SOTA baseline reproduction (MSDeepAMR, Weis et al.)
- [x] AMRNet trained and evaluated on DRIAMS-A
- [x] Cross-dataset validation on DRIAMS-B
- [x] ARO/CARD ontology interpretability pipeline
- [x] Genomic co-occurrence validation (CARD genome collection)
- [ ] Ablation studies
- [ ] Class imbalance mitigation for K. pneumoniae
- [ ] Manuscript preparation and submission

## Author

**Rajat Dwivedi**, B.Tech Information Technology, Global Institute of Technology, Jaipur.
Research internship under **Dr. Sarika Jain**, NIT Kurukshetra.

## License

*(Add your chosen license here, for example MIT for the code. DRIAMS and CARD each carry their own data-use licenses, which apply regardless of this repository's code license.)*
