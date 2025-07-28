# Deep learning to predict left ventricular hypertrophy from the electrocardiogram

Scripts for training and evaluation of AI-ECG for left ventricular (LV) mass regression and LV hypertrophy classification.

Citation:
```
Naderi, H., Kaplan, T. et al.  Deep learning to predict left ventricular hypertrophy from the electrocardiogram. In review (2025).
```

Bibtex:
```
Pending...
```

This research utilised Queen Mary University of London's Apocrita HPC facility, supported by QMUL Research-IT (http://doi.org/10.5281/zenodo.438045).


## Requirements

- Python v3.11.11
- PyTorch v2.6.0 (CPU)
- For other dependencies, see `environment.yml`
- Datasets require separate application and approval:
    - UK Biobank (UKB, https://www.ukbiobank.ac.uk/)
    - Study of Health in Pomerania (SHIP, https://ds-mica.qihs.uni-greifswald.de/study/ship)

_Verified on Rocky Linux v9.4 using Conda Miniforge v24.7.1_

## Pre-processing

12-lead ECGs and clinical variables from UKB and SHIP were pre-processed using proprietary processing scripts, as described in the manuscript. Please get in touch for assistance.

## Usage

1. Update UKB/SHIP data paths in `libs/ecg-ukb.py` as necessary.
2. Model training, see `python train.py --help` (examples below).
3. Downstream analyses, see `notebooks/{analysis,supervised-benchmark}.ipynb`.

Example usage:
```bash
# Train FCN (pre-defined hyperparameters) to predict indexed LVM (iLVM) from UKB annotations
time python train.py --verbose --save "UKB_iLVM" --target "INDEXED_MASS" --train 
# Train FCN model to predict iLVM using just the ECG, i.e. excluding clinical variables
time python train.py --verbose --save "UKB_LVH_ecgonly" --target "LVH" --train --excl_meta
# Train ResNet34 (pre-defined hyperparameters from Soto et al., 2022) instead of an FCN
time python train.py --verbose --save "UKB_LVH_r34" --soto2022 resnet34 --target "LVH" --train
# Evaluate a pre-trained model and save performance statistics
time python train.py --save_results --verbose --target "INDEXED_MASS" --load UKB_iLVM.202503.pth 
# Fine-tune a pre-trained model on SHIP 
time python train.py --save_results --verbose --target "INDEXED_MASS" --load UKB_iLVM.202503.pth --train --ext_tune --save UKB_iLVM.202506.SHIP
```


