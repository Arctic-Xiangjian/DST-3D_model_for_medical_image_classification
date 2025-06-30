# Depth-Sequence Transformer (DST) for Segment-Level ICA Calcification Mapping on Non-Contrast CT

[![Paper-License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python-Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch-Version](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation for our paper: **"Depth-Sequence Transformer (DST) for Segment-Level ICA Calcification Mapping on Non-Contrast CT"**.

Our work introduces the Depth-Sequence Transformer (DST), a novel framework that successfully addresses the previously intractable challenge of automated, segment-level calcification mapping. By reformulating the 3D localization problem as a 1D sequence analysis task, our method can achieve exceptional accuracy where conventional 3D deep learning models fail.

## 🏆 Usage as a General-Purpose 3D Backbone

Beyond its primary localization task, the DST architecture has proven to be a powerful and efficient general-purpose backbone for 3D volumetric classification. 

As demonstrated by our state-of-the-art results on the Clean-CC-CCII benchmark, its unique design offers an excellent trade-off between high performance and low computational cost (see Table III in our paper). This makes it a compelling alternative to larger, more resource-intensive models for various 3D analysis tasks. 

We are actively working on packaging the DST model and hope to contribute it to public model libraries to facilitate wider adoption and research.


## 📦 Dataset Preparation

Our framework is evaluated on two datasets.

### In-house NCCT Dataset (for ICA Landmark Localization)
Due to patient privacy regulations and IRB restrictions, our in-house clinical dataset cannot be made publicly available. However, our code is designed to work with data in a simple `.npy` format. **While the raw data cannot be shared, we welcome academic collaboration on multi-center validation studies.** Please contact the corresponding author for potential research collaborations.
### 2. Clean-CC-CCII (for External Validation)
The public Clean-CC-CCII dataset can be downloaded from its original source. After downloading, you can use our `dataset/preprocess.py` script to convert it into the `.npy` format required by our data loader.

**Pre-processing command:**
```bash
python dataset/preprocess.py --raw-root /path/to/your/raw/data --out /path/to/save/processed/npy_files
```

---

## 🚀 Training

All training and evaluation logic is handled by the main script `verify_train128.py`. Our `CT3DDataset` class handles the patient-level 5-fold cross-validation splits internally.

To train a model for a specific fold (e.g., fold 0), run the following command:

```bash
python verify_train128.py \
    --root /path/to/processed/npy_files \
    --fold 0 \
    --device cuda:0 \
    --batch 16 \
    --lr 2e-3 \
    --epochs 300 \
    --patience 50
```

---

## 📊 Evaluation
The evaluation is performed automatically at the end of the training script (`verify_train128.py`) on the test set of the corresponding fold. The script will load the model with the best validation performance and compute the final test metrics.

---

## 🔥 Ablation Studies
Our paper includes several ablation studies to validate our design choices. You can reproduce these by modifying the model architecture in `model/model128int.py` and re-running the training script. For example, to test the impact of the number of DST layers, you can change the `num_layers` argument in the `ResNet3D` initialization.

---

## 📈 Results
Our method achieves state-of-the-art performance on both the primary localization task and the external classification benchmark.

**ICA Landmark Localization (In-house Dataset):**
| Metric | Full Model (Ours) |
| :--- | :---: |
| MAE (slices) ↓ | **0.1** |
| Top-1 Acc (\%) ↑ | **93.12** |
| Acc$_{\tau=1}$ (\%) ↑ | **95.97** |

**3D Classification (Clean-CC-CCII):**
| Model | Task | AUC (↑) | F1 (↑) |
| :--- | :--- | :--- | :--- |
| Diff3Dformer* | 2-Class | (0.91) | (0.84) |
| **DST (Ours)** | **3-Class** | **0.99**| **0.94**|
> \* Result from original paper on a simplified task.

As shown in Table, our DST architecture establishes a new state-of-the-art (SOTA) on the Clean-CC-CCII benchmark. It is important to note that this superior performance was achieved by training our \textbf{general-purpose framework} directly, without the use of task-specific training techniques (e.g., contrastive pre-training or complex data augmentation) often employed in papers specifically designed for chest CT classification. This result strongly validates the fundamental power and efficiency of our architectural design.

We believe this establishes our DST as a highly competitive baseline for 3D volumetric analysis. We are actively working to expand this benchmark with more diverse datasets and are currently seeking publicly available cohorts suitable for validating the framework's dual-task capabilities in simultaneous localization and classification.

---

## 📜 Citation
The paper is currently pending submission, the pre-print will come to arxiv soon.

