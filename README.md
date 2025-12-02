# Subcellular Localization Prediction of Human Proteins Using Sequence-Based Machine Learning Models

Bioinformatics Project  
Authors: Sejal Khade, Aashay Deshpande, Pratyush Ingale, Krishna Shete

---

## 1. Project Overview

This repository contains a complete, end-to-end machine learning pipeline for predicting the subcellular localization of human proteins. Using the DeepLoc-1.0 dataset of 14,004 protein sequences across 10 localization classes, we explore two families of features:

1. Amino Acid Composition (AAC)  
2. ESM-2 Transformer Embeddings (facebook/esm2_t6_8M_UR50D)

We implement, train, and evaluate several classical machine learning models, perform hyperparameter tuning, and generate a comprehensive evaluation dashboard. The entire pipeline runs inside a single Google Colab notebook for full reproducibility.

---
##⚠️ Note on GitHub Notebook Rendering

**The notebook contains interactive Plotly dashboards and widget metadata, which GitHub cannot render.
If the notebook does not display on GitHub, please download it and open it in Google Colab for full functionality**.

---

## 2. Dataset Summary

- **Source:** DeepLoc-1.0 dataset (DTU HealthTech)  
- **Total sequences:** 14,004  
- **Classes:** 10 subcellular compartments  
- **Feature types:**  
  - AAC (20-dimensional)  
  - ESM-2 embeddings (320-dimensional)

The dataset is highly imbalanced, with nucleus and cytoplasm dominating the distribution. Minority classes such as plastid, peroxisome, and lysosome contain fewer than 400 samples. Metrics such as macro-averaged F1-score are therefore emphasized.

---

## 3. Methods and Workflow

### Week 1: AAC Baseline Models
- Parsed FASTA files using Biopython  
- Extracted 20-dimensional AAC feature vectors  
- Trained Random Forest, SVM, and MLP classifiers  
- Observed baseline accuracies around 0.54–0.56  
- Concluded that AAC lacks positional/motif-based features

### Week 2: ESM-2 Transformer Embeddings
- Loaded pretrained ESM-2 model (`facebook/esm2_t6_8M_UR50D`)  
- Generated 320-dimensional embeddings for all proteins  
- Trained RF, SVM, and MLP on embeddings  
- Accuracy improved significantly (0.69–0.75 range)

### Week 3: Hyperparameter Tuning and Visualization
- Applied RandomizedSearchCV  
- Best SVM parameters:
  - C ≈ 12.9  
  - gamma ≈ 0.00215  
- Evaluated tuned SVM using:
  - accuracy  
  - macro-F1  
  - confusion matrix  
  - ROC and PR curves  
  - per-class metrics  
  - UMAP embedding visualization  
- Created a unified Plotly dashboard combining all results

---

## 4. Final Model Performance

### Accuracy Comparison (AAC vs ESM Models)

| Model             | Accuracy               |
|-------------------|------------------------|
| RF (AAC)          | 0.56                   |
| SVM (AAC)         | ~0.55                  |
| MLP (AAC)         | ~0.54                  |
| RF (AAC + SMOTE)  | ~0.58                  |
| RF (ESM)          | 0.694                  |
| SVM (ESM)         | 0.748                  |
| MLP (ESM)         | 0.735                  |
| RF (ESM, Tuned)   | ~0.72                  |
| SVM (ESM, Tuned)  | 0.76–0.80 (Best Model) |

### Macro-F1 (ESM Models)

| Model                | Macro-F1   |
|----------------------|------------|
| RF (ESM)             | ~0.60      |
| SVM (ESM)            | ~0.68      |
| **SVM (ESM, Tuned)** | **~0.70+** |

The tuned SVM using ESM embeddings consistently outperformed all other models.

---

## 5. Visualization Dashboard

The notebook generates a complete Plotly dashboard including:

1. UMAP projection of ESM embeddings  
2. Confusion matrix (tuned SVM)  
3. Accuracy comparison bar chart  
4. Macro-F1 comparison (ESM models)  
5. Macro-averaged ROC curve  
6. Macro-averaged Precision–Recall curve  
7. Per-class precision, recall, and F1-score chart  

All plots are interactive and optimized for analysis.

---

## 6. Biological Interpretation

- Extracellular proteins cluster clearly due to strong N-terminal signal peptides.  
- Nuclear proteins show strong separability due to known localization motifs.  
- Mitochondrial localization is supported by transit peptide structure captured by ESM embeddings.  
- Peroxisome, plastid, and Golgi proteins remain challenging due to limited representation and subtle signals.  
- ESM embeddings significantly enhance model performance by capturing sequence order, motifs, and structural patterns missing in AAC.

---

## 7. Running the Project (Google Colab)

1. Open the notebook `protein_localization_ml_pipeline.ipynb` in Google Colab.  
2. Upload the dataset (`deeploc_data.fasta` or `deeploc_data.csv`).  
3. Change runtime to **GPU** (A100 or T4 recommended).  
4. Run all cells.  
5. The notebook will:
   - parse the data  
   - compute AAC and ESM features  
   - train and tune ML models  
   - generate evaluation metrics  
   - save trained models  
   - display the full interactive dashboard  

---

## 8. Saved Models

The following models and utilities are saved automatically:

- `best_svm_esm_tuned.joblib`  
- `rf_emb.joblib`  
- `label_encoder.joblib`  
- `scaler_emb.joblib`  
- `scaler_aac.joblib`  

These can be reused for downstream inference or deployment.

---

## 9. References

- Almagro Armenteros et al., “DeepLoc: Prediction of Protein Subcellular Localization Using Deep Learning” (2017).  
- Rives et al., “Biological Structure and Function Prediction with ESM Transformer Models.”  
- DTU HealthTech – DeepLoc Dataset.  
- scikit-learn documentation.

---

## 10. Contributors and Roles

- **Sejal Khade** — Biological analysis, dataset interpretation  
- **Aashay Deshpande** — AAC pipeline, preprocessing, initial EDA  
- **Pratyush Ingale** — ESM embeddings, model evaluation, dashboard development, documentation  
- **Krishna Shete** — Workflow organization, tuning, reproducibility planning  

---

