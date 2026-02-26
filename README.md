# POD-FM

> **Note:** This repository is reserved for the open-source code of our research work "Flow matching with gappy proper orthogonal decomposition guidance for balanced global field reconstruction".

## 📖 Overview

**POD-FM** is a research framework that combines **Proper Orthogonal Decomposition (POD)** with **Flow Matching (FM)** for field reconstruction. The framework supports multiple benchmark cases including **Pipe flow**, **Cylinder flow**, and **SST (Sea Surface Temperature)** scenarios, with both FM and POD-FM versions.

---

## 📁 Repository Structure

```
POD-FM/
├── data/                        # Raw data directory
├── dataset/                     # Processed dataset directory
├── model/                       # Model definition directory
├── train_Pipe.py                # Train FM model on Pipe case
├── train_Pipe_8rank.py          # Train POD-FM model on Pipe case (8 POD modes)
├── train_Cylinder.py            # Train FM model on Cylinder case
├── train_Cylinder_3rank.py      # Train POD-FM model on Cylinder case (3 POD modes)
├── train_SST.py                 # Train FM model on SST case
├── train_SST_rank3.py           # Train POD-FM model on SST case (3 POD modes)
├── sample_Pipe.py               # Inference with FM model on Pipe case
├── sample_Pipe_8rank.py         # Inference with POD-FM model on Pipe case (8 POD modes)
├── sample_Cylinder.py           # Inference with FM model on Cylinder case
├── sample_Cylinder_3rank.py     # Inference with POD-FM model on Cylinder case (3 POD modes)
├── sample_SST.py                # Inference with FM model on SST case
├── sample_SST_3rank.py          # Inference with POD-FM model on SST case (3 POD modes)
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- PyTorch 2.5.1
- NumPy, SciPy, and other standard scientific computing libraries

### Dowmload dataset

Download the data from Link **https://drive.google.com/drive/folders/1IG6hxv6b20gBhDsZ4U766YqSLGF8iF3X?usp=sharing** to the data directory.

### Usage

We use the **Pipe flow** case as a running example. The workflow is the same for other cases (Cylinder, SST).

#### 1. Train the baseline FM model

```bash
python train_Pipe.py
```

#### 2. Test / Sample with the baseline FM model

```bash
python sample_Pipe.py
```

#### 3. Train the POD-FM model (POD-enhanced)

```bash
python train_Pipe_8rank.py
```

#### 4. Test / Sample with the POD-FM model

```bash
python sample_Pipe_8rank.py
```

---

### Other Cases

| Case     | Train (FM)           | Sample (FM)           | Train (POD-FM)           | Sample (POD-FM)           |
|----------|----------------------|-----------------------|--------------------------|---------------------------|
| Pipe     | `train_Pipe.py`      | `sample_Pipe.py`      | `train_Pipe_8rank.py`    | `sample_Pipe_8rank.py`    |
| Cylinder | `train_Cylinder.py`  | `sample_Cylinder.py`  | `train_Cylinder_3rank.py`| `sample_Cylinder_3rank.py`|
| SST      | `train_SST.py`       | `sample_SST.py`       | `train_SST_rank3.py`     | `sample_SST_3rank.py`     |

---

## 📄 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file included in this repository.

---

## 📬 Citation
```bash
@article{wang2026flow,
  title={Flow matching with gappy proper orthogonal decomposition guidance for balanced global field reconstruction},
  author={Wang, Xin and Li, Dali and Zhang, Laiping and Deng, Xiaogang},
  journal={Physics of Fluids},
  volume={38},
  number={2},
  year={2026},
  publisher={AIP Publishing}
}
```
