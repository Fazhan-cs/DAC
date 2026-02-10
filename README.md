# Decomposing and Composing: Towards Efficient Vision-Language Continual Learning via Rank-1 Expert Pool in a Single LoRA

This code repository contains the official implementation of the paper **"Decomposing and Composing: Towards Efficient Vision-Language Continual Learning via Rank-1 Expert Pool in a Single LoRA"** — accepted for **Oral Presentation** at the 40th AAAI Conference on Artificial Intelligence (AAAI 2026), Singapore (January 20-27, 2026).

## 1. Environment Setup
### 1.1 Requirements

- Python ≥ 3.8
- CUDA ≥ 11.3 (for GPU acceleration)
- Conda ≥ 4.12.0 (recommended for environment management)

### 1.2 Create & Activate Environment

Use the provided `environment.yml` to build a consistent environment:

```bash
conda env create -f environment.yml
conda activate your_env  
```
## 2. Data preparation
### 2.1 Target Datasets

Experiments use standard vision-language incremental learning datasets (aligned with MTIL benchmarks):

Aircraft, Caltech101, CIFAR10, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397

Put them under /your_path/data and change the path in the bash file:

 ```data_loc=/your_path/data```



## 3. Usage Instructions

Key directory layout:

```bash
mtil/
├── scripts/          # Training/testing/preprocessing scripts
├── src/              # Core model & training logic
│   ├── model/        
│   │   ├── alora.py/      # core
│   ...    
├── ...        

```

1. Navigate to the project directory:
```bash
cd mtil
```

2. Training and testing commands:
```bash
# Training command
bash scripts/train/train.sh

# Testing command
bash scripts/test/test.sh
```

## 4. Reproducibility Notes

- Experiments were run on a single A6000 (48GB); 
- Use random seed `42` 



## 5. Citation

If you find this work useful, please cite our paper:

```BIBTEX
@inproceedings{zhan2026decomposing,
  title={Decomposing and Composing: Towards Efficient Vision-Language Continual Learning via Rank-1 Expert Pool in a Single LoRA},
  author={Zhan, Fa and [Other Authors]},
  booktitle={Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026},
  publisher={AAAI Press}
}
```





## 6. Acknowledgements

This work builds on contributions from open-source projects:

- [MoE-Adapters](https://github.com/JiazuoYu/MoE-Adapters4CL)
- [ZSCL](https://github.com/Thunderbeee/ZSCL)

We sincerely thank these researchers for their excellent work.