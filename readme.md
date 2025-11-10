# Multimodal Palpation Sensing for Precise Prediction of Breast Lump Hardness and Size

## Overview

Early detection of breast lumps requires precise characterization of lump size and hardness beyond binary classification. This project introduces:

1. **PalpationDataset**: A multimodal force–motion dataset capturing tactile signals from silicone phantoms and porcine tissue. *(Dataset will be released later.)*  
2. **PalpationMM**: A deep learning framework integrating parallel multi-scale temporal convolutions, content–style disentanglement, and gated fusion for robust lump characterization.

This repository provides the code to facilitate further research in tactile breast screening.

## PalpationMM

PalpationMM is a **deep learning framework** for predicting lump properties:

1. **Parallel Multi-Scale Temporal Convolutions**  
   - Depthwise dilated convolutions for capturing multi-scale temporal patterns  
   - Squeeze-and-excitation for channel reweighting

2. **Content–Style Disentanglement**  
   - Separates subject-invariant content from subject-specific style  
   - Orthogonality regularization and adversarial learning remove examiner-specific cues

3. **Gated Fusion and Refinement**  
   - Aligns content features from force and motion modalities  
   - Gated fusion produces a unified feature representation  
   - Residual MLP predicts lump properties (binary detection, hardness, size)

## PalpationDataset



