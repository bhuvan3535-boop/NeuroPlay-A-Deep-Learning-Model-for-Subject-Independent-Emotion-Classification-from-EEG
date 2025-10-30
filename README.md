# NeuroPlay-A-Deep-Learning-Model-for-Subject-Independent-Emotion-Classification-from-EEG

This project is a complete, end-to-end pipeline for classifying human emotion from raw EEG (brainwave) signals using a 1D Convolutional Neural Network (CNN).
The primary goal was to build a subject-independent model—a model that can accurately predict the emotional state of a person it has never seen before. This model achieves a 86.56% mean accuracy across 28 subjects from the GAMEEMO dataset.

**The Problem: Why Is This Hard?**
Classifying EEG signals is notoriously difficult due to high inter-subject variability. In simple terms, every person's brain is different.
A model trained on one person's "calm" signal will likely fail on another's. This was validated with a baseline model:
Baseline Model: A Random Forest classifier with handcrafted Power Spectral Density (PSD) features.
The "Failure": While this model achieved 94.1% accuracy when trained and tested on the same person (subject-specific), its accuracy plummeted to 39.26% when tested on new, unseen subjects (subject-independent).
This 39% result proves that a simple model doesn't learn "emotion"; it just learns to recognize a specific person's brain patterns.

**The Solution: A 1D-CNN**
To solve this, I built a 1D-Convolutional Neural Network (CNN). Instead of being fed handcrafted features, this model learns the optimal features directly from the raw, 3-second EEG signal segments.
The 1D-CNN learns to identify the underlying temporal patterns of emotion that are common across all 28 subjects, making it a truly generalized and robust solution.

**Final results**
The 1D-CNN was trained and evaluated using a rigorous Leave-One-Group-Out (LOGO) cross-validation. This means 28 separate models were trained, where each model was trained on 27 subjects and tested on the 1 subject it had never seen.
Model                     Validation Method           Mean Accuracy
Random Forest (Baseline)  Subject-Independent (LOGO)  39.26%
1D-CNN (Final Model)      Subject-Independent (LOGO)  86.56%

**Core Technologies Used**
Python 3.x
Google Colab (for GPU acceleration)
TensorFlow / Keras: For building and training the 1D-CNN.
MNE-Python: The core library for EEG signal processing and epoching.
Scikit-learn: For the LOGO cross-validation, StandardScaler, and baseline RandomForestClassifier.
Pandas & NumPy: For data manipulation.
Seaborn & Matplotlib: For visualizing the final confusion matrix.
