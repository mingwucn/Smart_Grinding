# System Architecture: Smart Grinding

## High-Level Data Flow
1. **Raw Data**: AE and Vibration signals (TDMS/Bin).
2. **Preprocessing (`ReadData.ipynb`, `Signal_extraction.ipynb`)**:
   - Signal cleaning.
   - STFT / Mel-Spectrogram generation.
   - Time/Frequency feature extraction.
3. **Physics Integration (`Physical_informed.ipynb`)**:
   - Calculation of BDI and Thermal Severity ($S_t$).
   - Embedding physical parameters into the feature space.
4. **Dataset Creation (`GrindingData.py`)**:
   - Windowing and normalization.
   - Multi-modal dataset assembly.
5. **Training (`trainer.py`, `MyModels.py`)**:
   - Model selection (GRU-Attention, PA-TFT).
   - Training with combined MSE + Physics-Informed loss.
6. **Evaluation (`postprocessing/`)**:
   - Accuracy metrics.
   - Plot generation for papers.
7. **Explainability (`XAI_ModelWrapper.py`, `MyShap.py`)**:
   - Feature importance and temporal attention analysis.

## Core Components
- **Data Handler**: Manages loading and batching of multi-sensor data.
- **Model Zoo**: Contains various architectures (Ablation study variants).
- **Physics Engine**: Calculates domain-specific indicators (BDI, $S_t$).
- **Post-Processor**: Handles metric calculation and high-resolution visualization.

## Technical Stack
- **Framework**: PyTorch.
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, SHAP.
- **Format**: Python scripts for core logic, Jupyter Notebooks for exploration.
