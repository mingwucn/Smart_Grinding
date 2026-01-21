# Tech Stack: Smart Grinding

## Core Technologies

### Programming Languages
- **Python 3.10+**: Primary language for the entire pipeline.

### Machine Learning & AI
- **PyTorch**: Deep learning framework for GRU-Attention and PA-TFT models.
- **SHAP**: For explainable AI and feature importance analysis.
- **Scikit-learn**: For traditional ML metrics and utilities.

### Data Processing & Analysis
- **Pandas & NumPy**: Core data manipulation.
- **SciPy**: For signal processing.
- **nptdms**: For reading high-frequency sensor data.
- **PyWavelets**: For wavelet transform feature extraction.
- **librosa**: For audio/sensor signal analysis.
- **Dask**: For handling large-scale sensor data processing.

### Visualization
- **Matplotlib & Seaborn**: Basic and statistical plotting.
- **SciencePlots**: For publication-quality plot styles.
- **OpenCV**: For any image-based data processing (spectrograms).

## Research Tools

### Documentation & Publication
- **LaTeX**: Primary tool for manuscript preparation (Contextual Grinding, Grinding Fusion).
- **Markdown**: For project documentation and reports.

### Environment & Development
- **Jupyter Notebooks**: For exploratory data analysis and physical model development.
- **Git**: Version control.

## Environment Setup (Summary from README)
```bash
# Core dependencies
pip install numpy scipy natsort matplotlib seaborn Pillow opencv-python dask tqdm pandas scienceplots librosa scikit-learn nptdms zmq jupyter openpyxl PyWavelets dill einops

# PyTorch (with CUDA 12.6 support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
