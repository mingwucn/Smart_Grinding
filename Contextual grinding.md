1. Process-Setting Classifier (RQ1)
Purpose: Validate that AE/vibration signals inherently encode contextual information (process settings).
Model Type: Supervised classifier (1D CNN, LSTM, or Transformer).
Input:

Raw AE/vibration time series (windowed segments, e.g., 1-second intervals).
Output:
Probability distribution over 16 process setting combinations (multi-class classification).
Training:
Train on raw signals with process settings as labels.
Use stratified k-fold cross-validation.
2. Adversarial Generator (RQ2)
Purpose: Generate signals that fool the classifier (remove context) while preserving roughness-relevant dynamics.
Model Type: Conditional GAN or Adversarial Autoencoder.
Components:

Encoder: Maps raw signals to latent space ZZ.
Decoder: Reconstructs signals from ZZ.
Input:
Raw AE/vibration signals.
Output:
Adversarially perturbed signals (context-stripped).
3. Discriminators (RQ2)
3a. Classifier Discriminator
Purpose: Force the generator to remove contextual information.
Input:

Adversarial signals from generator.
Output:
Process setting predictions (same as RQ1 classifier).
3b. Predictor Discriminator
Purpose: Ensure latent features ZZ retain dynamics linked to roughness.
Input:

Latent features ZZ from generator.
Output:
Surface roughness prediction (R_aR 
a
​
  value).
4. Roughness Predictor (RQ2/RQ3)
Purpose: Quantify how well adversarial latents predict roughness.
Model Type: Regression model (MLP, XGBoost).
Input:

Latent features ZZ from adversarial generator.
Output:
Predicted surface roughness (R_aR 
a
​
 ).
Training:
Train on adversarial latents, test on held-out settings.
5. Latent Space Analyzer (RQ3)
Purpose: Evaluate if latents generalize across settings.
Models:

PCA/t-SNE: Reduce latents to 2D/3D for visualization.
Variance Calculator: Compare intra/inter-setting variance.
Cross-Setting Regressor: Train on latents from 12 settings, test on 4 unseen.
Input:
Latent features ZZ.
Output:
Clustering metrics, variance ratios, MAE for generalization.
6. Baseline Models (Ablation Studies)
Purpose: Compare against non-adversarial approaches.
6a. Autoencoder (AE)

Input: Raw signals.
Output: Reconstructed signals + non-adversarial latents.
6b. Traditional ML (e.g., Random Forest)
Input: Handcrafted features (e.g., RMS, FFT peaks).
Output: Roughness prediction.
Summary of Models
Model	Type	Input	Output	Role
Process-Setting Classifier	CNN/LSTM	Raw signals	Process setting labels (16 classes)	Validate context encoding (RQ1)
Adversarial Generator	GAN/Autoencoder	Raw signals	Adversarial signals	Strip context (RQ2)
Classifier Discriminator	Frozen Classifier	Adversarial signals	Process setting labels	Adversarial training (RQ2)
Predictor Discriminator	MLP Regressor	Latent features ZZ	Roughness R_aR 
a
​
 	Retain dynamics (RQ2/RQ3)
Roughness Predictor	MLP/XGBoost	Latent features ZZ	Predicted R_aR 
a
​
 	Quantify dynamics (RQ3)
PCA/t-SNE	Dimensionality Reduction	Latent features ZZ	2D/3D projections	Visualize invariance (RQ3)
Autoencoder (Baseline)	AE	Raw signals	Reconstructed signals + latents	Non-adversarial comparison
Traditional ML (Baseline)	RF/SVM	Handcrafted features	Predicted R_aR 
a
​
 	Benchmark performance
Key Design Notes
Adversarial Training Workflow:
Freeze the RQ1 classifier after training.
Jointly train the generator and predictor discriminator, using the frozen classifier as an adversary.
Input Normalization:
Standardize AE/vibration signals per sensor channel.
Signal Segmentation:
Use sliding windows (e.g., 1024 samples/window) with 50% overlap for time-series processing.
Total Models
Core Models: 5 (Classifier, Generator, 2 Discriminators, Predictor).
Baselines: 2 (Autoencoder, Traditional ML).
Analytical Tools: 2 (PCA/t-SNE, Variance Analyzer).
