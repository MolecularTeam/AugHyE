# AugHyE
Official implementation of "AugHyE: generated structure augmentation with hybrid encoder for robust protein binding interface prediction."

**TL;DR:** We propose AugHyE, a novel framework that integrates generated structure augmentation with a Hybrid Encoder to enhance model robustness to structural variations. 

## Abstract
Protein binding interface (PBI) prediction is essential for elucidating biological mechanisms and accelerating drug discovery. Recent deep learning methods have achieved substantial performance improvements in identifying residues involved in protein interactions. However, their performance often degrades when applied to unbound structures, since these models are typically trained primarily on native bound structures and may not generalize well to structural variations. To address this limitation, we propose AugHyE, a novel framework that integrates generated structure augmentation with a Hybrid Encoder to enhance model robustness by expanding the training distribution beyond native bound structures. Our approach leverages ESM3 to generate protein structures from pro- tein sequences and incorporates an alignment network to reduce spatial overlap and adjust the relative positioning between the independently generated ligand and receptor structures. These aligned generated structures are com- bined with native bound structures to construct a unified training dataset, which is used to train the Hybrid Encoder that integrates local geometric features with global struc- tural context. We evaluate AugHyE on two PBI prediction benchmarks and achieve strong performance across multiple structural test settings, supporting its robustness to structural variation.


## Conda activation:
A Conda virtual environment setup will be available.

## Notes on Reproducibility (Dependencies)
This environment is tested with the following **key packages 
- python 3.10 
- pyTorch 2.1.0 (CUDA 11.8)
- torch_geometric 2.6.1 (CUDA 11.8)
- dgl 2.4.0 (CUDA 11.8)
- mamba-ssm 2.2.2
- numpy 1.23.5
- e3nn 0.5.0
- fair-esm 2.0.0
