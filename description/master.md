# Smart Grinding - Project Definition

## Idea
The Smart Grinding project develops advanced machine learning models for real-time monitoring and prediction of grinding process quality. It specifically focuses on predicting surface roughness ($R_a$) by fusing multi-sensor data (Acoustic Emission and Vibration) with physics-informed constraints.

## Core Features

### 1. Multi-Sensor Data Fusion
- **AE Monitoring**: Processing Acoustic Emission signals (spectrograms and statistical features).
- **Vibration Monitoring**: Processing vibration signals to capture machine dynamics.
- **Physical Parameter Integration**: Incorporating machining parameters like feed rate, depth of cut, and wheel speed.

### 2. Physics-Informed Learning
- **BDI Integration**: Using the Brittle-Ductile Indicator to inform the model about material removal regimes.
- **Thermal Severity ($S_t$)**: Monitoring thermal impacts to prevent grinding burn and improve prediction accuracy.
- **Constrained Optimization**: Incorporating physical laws into the loss function to ensure realistic predictions.

### 3. Explainable AI (XAI)
- **SHAP Analysis**: Quantifying the contribution of each sensor feature to the $R_a$ prediction.
- **Attention Visualization**: Highlighting critical temporal segments in the sensor data that correlate with surface quality.

## Technical Scope

### Input Formats
- High-frequency sensor data (TDMS or binary formats).
- Extracted features (Time-domain, Frequency-domain, Wavelet).
- Machine controller parameters.

### Processing Capabilities
1. **Signal Preprocessing**: Noise reduction, normalization, and STFT/Mel-spectrogram generation.
2. **Feature Engineering**: Automated and manual extraction of grinding-specific features.
3. **Deep Learning**: Training GRU-Attention and Physics-Aware Transformers on sequence data.

### Codebase Structure
- **Model Definition (`MyModels.py`)**: Contains the core PyTorch model architectures (e.g., GrindingPredictor, FeatureInterpreter).
- **Data Loading (`MyDataset.py`)**: Handles all data loading logic, dataset construction, and batching. Optimized for performance.
- **Training Loop (`trainer.py`)**: Orchestrates the training process, including 10x10 Cross-Validation, loss calculation, and optimization.
- **Accuracy Analysis (`postprocessing/generate_accuracy_report.py`)**: Script used to evaluate model performance, generate accuracy-related plots, and produce summary reports.
- **Physics Plotting (`postprocessing/generate_physical_regime_plots.py`)**: Generates high-fidelity time-series plots with physical regime context, supporting CUDA acceleration for full-dataset visualization.
- **Prediction Archival (`postprocessing/generate_all_predictions.py`)**: Stores prediction results (labels, predictions, physical indicators) for all models and folds into the `lfs/predictions/` directory, supporting resume functionality.

### Output Deliverables
1. **Prediction Models**: Robust models for $R_a$ estimation.
2. **Research Manuscripts**: Publication-ready LaTeX documents for "Grinding Fusion" and "Contextual Grinding".
3. **Visualization Suite**: Scripts for generating high-quality research plots.
4. **Prediction Data Archive**: Comprehensive storage of model predictions across 10x10 Cross-Validation for offline analysis.

## Use Cases

### Precision Manufacturing
- Real-time quality control in precision grinding operations to reduce scrap and rework.

### Process Optimization
- Identifying optimal machining parameters by analyzing the relationship between BDI, $S_t$, and surface quality.

## Success Criteria
1. **Accuracy**: Achieve low MAE/MSE across 10x10 Cross-Validation.
2. **Robustness**: Maintain performance across different physical regimes (ductile vs. brittle).
3. **Interpretability**: Provide physically meaningful explanations for model predictions.
