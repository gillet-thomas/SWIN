<div align="center">    
 
# SwiFT: Swin 4D fMRI Transformer

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.12+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
</div>

📌 **Introduction**

This project presents a custom implementation of [SwiFT (Swin 4D fMRI Transformer)](https://github.com/Transconnectome/SwiFT), a scalable analysis model for fMRI. SwiFT, based on the Swin Transformer, is designed to effectively predict various biological and cognitive variables from fMRI scans. This iteration of the project focuses on providing a lightweight and flexible framework for fMRI analysis, moving away from PyTorch Lightning to a streamlined PyTorch-only implementation.

We have significantly refactored the original codebase to offer a more direct and intuitive training experience with a custom training loop and a lighter file code architecture. This allows for greater control and easier integration into diverse research workflows.

You can find the original research paper on SwiFT [here](https://arxiv.org/abs/2307.05916). Feel free to contact the authors regarding the original project.

---

## 🚀 Key Features and Improvements

This custom implementation of SwiFT offers the following:

* **Pure PyTorch Implementation:** Removed all dependencies of PyTorch Lightning, resulting in a cleaner and easier codebase.
* **Custom Training:** the PyTorch Lightning-based training has been replaced with a simplified, custom training loop, providing a more direct and understandable approach to model training.
* **Streamlined Data Handling:** Simplified data preprocessing and loading pipelines for 4D fMRI datasets.
* **Optimized for CUDA Devices:** Easy and efficient utilization of CUDA GPUs for faster training and inference.
* **Advanced Visualization Tools:**
    * **t-SNE & UMAP Visualization:** Integrated t-Distributed Stochastic Neighbor Embedding (t-SNE) and Uniform Manifold Approximation and Projection (UMAP) for visualizing high-dimensional model predictions.
    * **Integrated Gradients:** Simplified implementation compared to the original project, making it easier to apply for model interpretability.
* **Contrastive Language–Image Pre-training:** Includes a CLIP contrastive learning implementation for aligning paired modalities (e.g., age-gender).

---

## 🧠 Experiments and Results

This repository includes implementations and evaluations of SwiFT on the ADNI dataset, preprocessed and aligned to MNI space. Various classification tasks were conducted:

1.  **Age Group Prediction:**
    * **Task:** Classifying individuals into "young" (< 69 years old) and "old" (> 78 years old) age groups.
    * **Performance:** Achieved 95.24% accuracy on the validation dataset.

2.  **Gender Prediction:**
    * **Task:** Classifying individuals based on their gender.
    * **Performance:** Achieved 93.34% accuracy on the validation dataset.

3.  **Four-Target Classification:**
    * **Task:** Classifying individuals into four distinct groups: Young Female, Young Male, Old Female, Old Male.
    * **Performance:** Achieved 89.2% accuracy on the validation dataset.

4.  **CLIP Version:**
    * **Task:** A CLIP-inspired version where the Swin model trained for age group prediction was aligned with one-hot encoded gender information.
    * **Performance:** Achieved 97.46% accuracy on the validation dataset.

5.  **AD vs. CN Classification (Alzheimer's Disease vs. Cognitively Normal):**
    * **Task:** Classifying individuals as either having Alzheimer's Disease (AD) or being Cognitively Normal (CN).
    * **Performance:** Achieved 97.2% accuracy on the validation dataset.

---

## 📊 Visualizations

We have integrated visualization capabilities to better understand the model's performance and the underlying data structure.

### t-SNE & UMAP Projections

These plots showcase the dimensionality reduction of fMRI features into 2D space, helping to visualize the separation of different classes.

**Placeholder for t-SNE Visualization:**

---

## 💻 Getting Started

While we've simplified the setup, basic knowledge of PyTorch is recommended for effective usage.

```bash
# Example: To run a training script (adjust as per your actual script names)
python train.py --config configs/my_config.yaml
