# Subcellular-Localization-Prediction-of-Human-Proteins-Using-Sequence-Based-Machine-Learning-Models-
Bioinformatics Project by:- Sejal Khade, Aashay Deshpande, Pratyush Ingale, Krishna Shet

# Protein Subcellular Location Classification

 # Project Summary (5-Line Overview)

This project aims to predict protein subcellular localization using machine learning on the DeepLoc dataset (14,004 sequences, 10 classes). We implemented a reproducible pipeline in Google Colab that extracts the simple Amino Acid Composition (AAC) features. We then trained and optimized three classifiers—Random Forest (RF), Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP)—and evaluated their performance using Macro Avg F1-score and Confusion Matrices to identify the best model for this classification task.

# How to Run the Project

This project is entirely contained within a single Google Colab notebook for maximum reproducibility.

Upload Data: Upload the deeploc_data.fasta file to your Google Colab environment.

Open Notebook: Open the project_notebook.ipynb file in Google Colab.

Set Runtime: Ensure your runtime is set to T4 GPU (Runtime > Change runtime type) to minimize the long training time required for the SVM model.

Run All Cells: Execute all cells sequentially (Runtime > Run all). The notebook will automatically install dependencies, parse the data, extract features, train all models with timing, and generate performance plots.

# 2. Data Summary

Data Source: DeepLoc-1.0 training dataset from DTU HealthTech

Dataset Size: 14,004 proteins across 10 distinct subcellular localization classes.

Key Finding: Significant class imbalance was observed, which the model evaluation accounts for using the Macro Avg F1-score.

# 3. Methods and Workflow

Feature Engineering: Amino Acid Composition (AAC), resulting in a 20-dimensional feature vector per protein.

Models: Random Forest (RF), Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP).

Evaluation: Macro Avg F1-score, Accuracy, Confusion Matrices.

# 4. Key Results Snapshot

Results will be updated upon completion of the model training and evaluation in Week 3.

Best Model: TBD

Macro Avg F1-Score: TBD


