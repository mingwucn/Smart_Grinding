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
4. **Dataset Construction (`MyDataset.py`)**:
   - Encapsulates all data loading and preprocessing logic.
   - Manages windowing, normalization, and batch creation.
5. **Training (`trainer.py`)**:
   - Instantiates models from `MyModels.py`.
   - Executes the training loop, managing epochs, validation, and physics-informed loss calculation.
6. **Evaluation (`postprocessing/`)**:
   - Accuracy metrics.
   - Plot generation for papers.

## Core Components
- **Data Handler (`MyDataset.py`)**: The single source of truth for loading and serving data to the model.
- **Model Definitions (`MyModels.py`)**: Defines the neural network architectures (GRU-Attention, PA-TFT).
- **Training Orchestrator (`trainer.py`)**: Manages the lifecycle of model training and validation.
- **Physics Engine**: Calculates domain-specific indicators (BDI, $S_t$).
- **Post-Processor**: Handles metric calculation and high-resolution visualization.

## Technical Stack
- **Framework**: PyTorch.
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, SHAP.
- **Format**: Python scripts for core logic, Jupyter Notebooks for exploration.
