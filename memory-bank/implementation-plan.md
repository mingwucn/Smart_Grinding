# Implementation Plan: Smart Grinding

## Phase 1: Data & Model Foundation (Completed)
- [x] Preprocessing scripts for AE and Vibration signals.
- [x] Physics-informed feature extraction (BDI, $S_t$).
- [x] Core model architectures (GRU-Attention, PA-TFT).
- [x] 10x10 Cross-Validation pipeline.

## Phase 2: Evaluation & Ablation Study (In Progress)
- [x] Run 10x10 CV for all specified input combinations (Ablation study).
- [x] Generate MAE/MSE summary reports.
- [ ] Implement Model Performance Across Physical Regimes visualization.
  - [ ] Panel A: Prediction vs. Ground Truth with BDI color-coding.
  - [ ] Panel B: Error Analysis vs. BDI scatter plot.
  - [ ] Panel C: Error Analysis vs. Thermal Severity ($S_t$) scatter plot.

## Phase 3: Interpretability & XAI (Planned)
- [ ] Execute SHAP analysis on best-performing models.
- [ ] Visualize Attention weights for key sequences.
- [ ] correlate XAI results with physical grinding regimes.

## Phase 4: Publication & Documentation (Planned)
- [ ] Finalize "Grinding Fusion" manuscript figures.
- [ ] Update LaTeX text for "Contextual Grinding".
- [ ] Clean up `utils/` and `postprocessing/` scripts for reproducibility.
- [ ] Final project summary and model export.

## Maintenance
- [ ] Continuous verification of new features using `tests/`.
- [ ] Update `memory-bank/progress.md` after each milestone.
