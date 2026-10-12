# AMRNet: Attention-Augmented Antimicrobial Resistance Prediction from MALDI-TOF Mass Spectra with Ontology-Grounded Interpretability

Official PyTorch implementation of **AMRNet**, an attention-driven deep learning framework designed to profile antimicrobial resistance (AMR) directly from intact-cell MALDI-TOF mass spectra. Evaluated on the clinical DRIAMS biobank across internal 5-fold cross-validation and external multi-center cohorts, with model interpretability grounded in the Comprehensive Antibiotic Resistance Database (CARD).

---

## Repository Structure

```text
AMRNet/
├── Datasets/
│   ├── DIRAMS.txt                           # Direct download links for DRIAMS cohorts
│   ├── card-data.tar.zip                    # CARD ontology reference definitions
│   ├── card-genomes.txt.gz                  # CARD whole-genome co-occurrence database
│   └── card_prevalence.txt.gz               # Population-level prevalence statistics
├── interpretability/
│   └── colab_ontology_interpretability.py   # Attention weight extraction & CARD gene matching
├── ontology/
│   └── 3_card_aro_ontology_lookup.py        # Constructs card_aro_lookup.csv (ARO mapping)
├── driam_a_processing.py                    # Complete DRIAMS-A pipeline: data prep & AMRNet model
├── driam_b_preprocessingAndValidation.py    # Multi-center external validation on DRIAMS-B
├── benchmark_sota_baselines.py              # Comparative benchmark (Weis RF, 1D-CNN) & DeLong tests
├── five_fold_cross_validation.py            # 5-fold stratified cross-validation suite
├── card_mass_tolerance_sweep.py             # Mass-tolerance sensitivity sweep (±1 to ±15 Da)
├── requirements.txt                         # Python runtime dependencies
├── LICENSE
└── README.md

```

(The repository structure incorporates the core preprocessing, ontology, and interpretability modules alongside benchmark evaluation scripts).

---

## Datasets

Due to file size constraints, raw spectra and genomic databases are hosted via public mirrors and archives:

| Dataset | Scope / Description | Access Link | Primary Role |
| --- | --- | --- | --- |
| **DRIAMS-A & B (Kaggle Mirror)** | Raw Bruker MALDI-TOF spectra (~15 GB compressed) | [Kaggle Dataset](https://www.kaggle.com/datasets/drscarlat/driams/data) | Direct download mirror |
| **DRIAMS (Official Biobank)** | DRIAMS-A ($n = 18{,}235$), DRIAMS-B ($n = 5{,}897$) | [Dryad Archive](https://doi.org/10.5061/dryad.bzkh1899q) | Primary cohort citation |
| **CARD Reference Database** | 1,297 species-matched AMR protein records | [CARD Portal](https://card.mcmaster.ca/download) | Reference lookup construction |
| **CARD Genomes & Prevalence** | 250,685 whole-genome annotations | [CARD Download](https://card.mcmaster.ca/download) | Population co-occurrence evaluation |

### Quick CLI Download

```bash
kaggle datasets download -d drscarlat/driams
unzip driams.zip -d ./data/

```

---

## Model Architecture

AMRNet processes mass spectra across 6,000 bins ($m/z$ 2,000–20,000 Da, 3 Da bin resolution):

* **Input-Level Attention Gate:** A parameter weight vector ($\mathbb{R}^{6000} \to \mathbb{R}^{6000}$) with Sigmoid activation learning per-bin saliency without altering physical $m/z$ coordinates.
* **Deep Representation Backbone:** Multi-layer perceptron ($1024 \to 256 \to 64$) equipped with Batch Normalization, ReLU activations, and Dropout ($p = 0.3$).
* **Imbalance Handling:** Inverse class frequency sampling via PyTorch `WeightedRandomSampler`.
* **Objective & Optimization:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`), Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$), and early stopping on validation AUROC.

---

## Experimental Results

### 1. Internal Validation: 5-Fold Stratified Cross-Validation (DRIAMS-A)

Evaluated across 5 stratified folds ($\mu \pm \sigma$) on clinical isolates from DRIAMS-A:

| Species / Antibiotic | Sample Size ($n$) | Prevalence ($\pi$) | AUROC | AUPRC | F1-Score | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| ***Escherichia coli* / Ciprofloxacin** | 4,891 | 28.8% | **0.820 ± 0.010** | **0.709 ± 0.010** | 0.610 ± 0.024 | 0.794 ± 0.005 |
| ***Klebsiella pneumoniae* / Ceftriaxone** | 2,848 | 15.3% | **0.831 ± 0.024** | **0.660 ± 0.064** | 0.561 ± 0.070 | 0.899 ± 0.013 |
| ***Staphylococcus aureus* / Oxacillin** | 3,791 | 19.2% | **0.893 ± 0.015** | **0.768 ± 0.029** | 0.655 ± 0.021 | 0.887 ± 0.009 |

---

### 2. SOTA Comparative Benchmark & Statistical Significance

Benchmarked against the clinical Random Forest ensemble of Weis et al. (*Nature Medicine* 2022) and a 1D Convolutional Neural Network baseline (*Frontiers in Microbiology* 2024) on identical held-out test partitions. Statistical significance is evaluated via paired DeLong tests:

| Cohort | 1D-CNN Baseline | Weis et al. (RF) | **AMRNet (Ours)** | DeLong $p$ (vs. 1D-CNN) | DeLong $p$ (vs. RF) |
| --- | --- | --- | --- | --- | --- |
| ***E. coli* / Ciprofloxacin** | 0.623 | 0.785 | **0.828** | $\mathbf{3.55 \times 10^{-15}}$ | $\mathbf{0.0307}$ |
| ***K. pneumoniae* / Ceftriaxone** | 0.640 | **0.822** | 0.803 | $\mathbf{2.40 \times 10^{-4}}$ | 0.6044 (n.s.) |
| ***S. aureus* / Oxacillin** | 0.681 | 0.860 | **0.885** | $\mathbf{3.53 \times 10^{-11}}$ | 0.2090 (n.s.) |

*Note:* The performance collapse of 1D-CNN reflects the failure of convolutional translation invariance in mass spectrometry, where shifting coordinates across bins disrupts physical molecular mass identities.

---

### 3. Multi-Center External Validation (DRIAMS-B)

Models trained on DRIAMS-A were evaluated without fine-tuning on external spectra from DRIAMS-B:

| Cohort | Raw AUROC | Filtered AUROC | Abstention Rate | Generalization Finding |
| --- | --- | --- | --- | --- |
| ***E. coli* / Ciprofloxacin** | 0.827 | **0.832** | 3.8% | Direct transferability across clinical centers |
| ***S. aureus* / Oxacillin** | 0.700 | **0.714** | 3.5% | Stable transfer amidst center acquisition variance |
| ***K. pneumoniae* / Ceftriaxone** | 0.421 | *Abstained* | — | Flagged by OOD filter due to physical marker absence |

---

### 4. Mass-Tolerance Sensitivity Analysis (CARD Grounding)

Evaluation of theoretical mass matching across tolerance windows ($\pm 1$ to $\pm 15$ Da) within the linear biotyping window (2,000–20,000 Da):

| Tolerance Window | Total CARD Matches | Unique Genes (Window) | Matched Bins | Cohort Salient Matches (*S. aureus* / *E. coli* / *K. pneum.*) |
| --- | --- | --- | --- | --- |
| **$\pm$ 1 Da** | 23 | 23 | 23 | 0 / 0 / 0 |
| **$\pm$ 3 Da** | 68 | 34 | 68 | 0 / 0 / 0 |
| **$\pm$ 5 Da** | 114 | 34 | 113 | 1 / 0 / 0 |
| **$\pm$ 15 Da** | 340 | 34 | 320 | 3 / 2 / 4 |

Linear positive MALDI-TOF instruments ($m/\Delta m \approx 500\text{--}1000$) exhibit natural peak broadening ($\pm 10\text{--}15$ Da) in the $10\text{--}20$ kDa range. The $\pm 15$ Da window matches physical spectrometer tolerances, uncovering biologically grounded surrogates such as plasmid-borne *qacJ* in resistant *S. aureus* (**Odds Ratio = 5.88**, $p < 10^{-15}$).

---

## Installation & Setup

```bash
git clone https://github.com/Rajat-Dwivedi63/AMR_Net-.git
cd AMR_Net-
pip install -r requirements.txt

```

> **Google Colab Users:** All required dependencies (`torch`, `scikit-learn`, `numpy`, `pandas`, `scipy`) are pre-installed in the Colab runtime environment.

---

## Execution Guide

### 1. Build CARD Reference Table

Constructs the theoretical molecular weight lookup table:

```bash
python ontology/3_card_aro_ontology_lookup.py

```

### 2. Run Main AMRNet Training & Internal Test Evaluation

Processes DRIAMS-A spectra, trains AMRNet with early stopping, and logs test metrics:

```bash
python driam_a_processing.py

```

### 3. Run 5-Fold Stratified Cross-Validation Suite

Generates fold metrics and outputs mean $\pm$ standard deviation scores:

```bash
python five_fold_cross_validation.py

```

### 4. Benchmark Competing SOTA Models & Statistical Significance

Trains Weis et al. Random Forest and 1D-CNN baselines, running paired DeLong tests:

```bash
python benchmark_sota_baselines.py

```

### 5. Multi-Center External Validation (DRIAMS-B)

Evaluates trained models on external clinical spectra with uncertainty abstention filtering:

```bash
python driam_b_preprocessingAndValidation.py

```

### 6. Biological Saliency & Mass-Tolerance Analysis

Extracts input attention saliency profiles and evaluates resolution bounds:

```bash
python card_mass_tolerance_sweep.py
python interpretability/colab_ontology_interpretability.py

```

---

## Citation

```bibtex
@article{dwivedi2026amrnet,
  title={AMRNet: Attention-Augmented Antimicrobial Resistance Prediction from MALDI-TOF Mass Spectra with Ontology-Grounded Interpretability},
  author={Dwivedi, Rajat and Jain, Sarika},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  note={Under Review}
}

```

## License

This project is licensed under the MIT License — see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details. Datasets are subject to their respective institutional biobank licenses (DRIAMS, CARD).
