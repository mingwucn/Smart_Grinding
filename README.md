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
- DTW.ipynb

## 
- Hierarchy.ipynb