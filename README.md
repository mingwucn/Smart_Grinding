---
tag: 
  - Manufacturing/Grinding
  - project/Smart_Grinding
type: git repository
setup: 
    - numpy scipy natsort matplotlib seaborn Pillow opencv-python dask tqdm pandas scienceplots librosa scikit-learn nptdms zmq jupyter openpyxl PyWavelets dill einops
    - torchvision torch torchaudio 
    - torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
---
---
Code for the Smart Grinding
## Subfolder
- [Paper] Grinding Fusion
```
git clone https://git@git.overleaf.com/67bc423e3e17261c3b4e1197 "Grinding Fusion"
```

- [Paper] Contextual Grinding
```
git clone https://git@git.overleaf.com/67b9c453358700f01df35699 "Contextual Grinding"
```

- utils
```
git clone git@github.com:mingwucn/utils.git "utils"
```

## Preprocessing
Read data: ``ReadData.ipynb``
Physical data embedded: ``Physical_informed.ipynb``
Signal feature extraction: ``Signal_extraction.ipynb``

```bash
python GrindingData.py --threads=6 --process_type=spec
python GrindingData.py --threads=6 --process_type=physics
```

## 
Code-Paper

###
Finalize Model Architecture:

- [x] Action: Ensure the Python code for your chosen model (e.g., the memory-efficient GRU-Attention model or a PA-TFT variant) is finalized, debugged, and matches the description you'll put in the paper.
- [x] Output: Clean, runnable Python code for the GrindingPredictor and its sub-modules.

- [x] Run/Re-run 10x10 Cross-Validation:

- [x] Action: Execute the 10x10 CV for all specified input combinations (ablation study).

'all' (full model)
'ae_spec_only'
'ae_features_only'
'ae_features+pp'
'vib_spec_only'
'vib_features_only'
'vib_features+pp'
Consider adding: 'pp_only', 'ae_all_only' (spec+features), 'vib_all_only' (spec+features), 'all_sensors_no_pp', 'full_model_no_St_loss'.


- [x] Output: MAE and MSE (mean and std dev) for each combination across all folds/repeats.



- [ ] Implement/Run Interpretability Analysis:
  - [ ] Action: Apply SHAP or extract attention weights from your best-performing model (and potentially key ablation models).
  - [ ] Output: SHAP value plots (summary plot, dependence plots for key features) or attention weight visualizations.



Physics Constraint Tracking (Optional but good):

- [ ] Action: If feasible, modify your training loop to log when the $S_t$ penalty term is active and its magnitude, or how BDI values correlate with prediction errors.
- [ ] Output: Data/statistics on the activation and impact of physics constraints.



Generate Key Visualizations:

- [ ] Action: Create plots from your results:
  - [ ] Predicted vs. Actual $R_a$ scatter plot for the best model.
  - [ ] Bar chart/table summarizing ablation study MAE/MSE.
  - [ ] SHAP summary plots / Attention maps.
  - [ ] Model architecture diagram (can be drafted in Mermaid then formally drawn).
  - [ ] Output: High-quality image files for figures.

## 