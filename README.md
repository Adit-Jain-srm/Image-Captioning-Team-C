<div align="center">

# Image Captioning

**Deep Learning Image-to-Text Generation**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-CNN-5C3EE8?logo=opencv)](https://opencv.org/)
[![NLP](https://img.shields.io/badge/NLP-Transformer-09A3D5)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Adit-Jain-srm/Image-Captioning-Team-C)](https://github.com/Adit-Jain-srm/Image-Captioning-Team-C)

*Generating natural language descriptions of images using encoder-decoder neural networks.*

</div>

---

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project aims to generate descriptive captions for images using deep learning techniques, combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTM). The project involves data preparation, feature extraction, model building, training, evaluation, and fine-tuning to achieve accurate and coherent image descriptions.

### Key Features
- **Encoder-Decoder Architecture**: CNN encoder (InceptionV3) + LSTM decoder with attention mechanism
- **MS COCO Dataset**: Trained on large-scale image-caption pairs
- **BLEU Score Evaluation**: Quantitative assessment of caption quality
- **End-to-End Pipeline**: From data preprocessing to model deployment

---

## 🎯 Project Goals

The primary objective is to build a deep learning model that can:
1. Understand visual content in images
2. Generate natural language descriptions
3. Produce contextually relevant and grammatically correct captions
4. Generalize well to unseen images

---

## 📚 Table of Contents

1. [Step-by-Step Process](#step-by-step-process)
2. [Detailed Sections](#detailed-sections)
3. [Installation & Setup](#installation--setup)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Team Information](#team-information)
7. [Results](#results)
8. [FAQs](#faqs)
9. [References](#references)

---

## 🔄 Step-by-Step Process

### 1. Data Preparation
- Collect and preprocess the MS COCO dataset (images and captions)
- Resize images to 299×299 pixels and normalize pixel values (0-1 range)
- Tokenize and pad captions for uniformity

### 2. Feature Extraction
- Use pre-trained InceptionV3 model to extract image features
- Remove classification layer and extract feature vectors (8×8×2048)
- Save extracted features for reuse

### 3. Caption Preparation
- Convert captions into numerical sequences using Keras Tokenizer
- Add special tokens `<start>` and `<end>` for caption boundaries
- Pad sequences to uniform length

### 4. Model Building
- Create encoder-decoder architecture with attention mechanisms
- Combine image features and caption sequences for training

### 5. Model Training
- Split data into training (80%) and validation (20%) sets
- Train using Adam optimizer with categorical cross-entropy loss
- Monitor performance with callbacks (checkpointing, early stopping, LR scheduler)

### 6. Model Evaluation
- Evaluate using BLEU-1 to BLEU-4 scores
- Manual qualitative checks on generated captions
- Test on new, unseen images

### 7. Fine-Tuning
- Experiment with hyperparameters (learning rate, batch size, LSTM units)
- Add dropout and data augmentation to prevent overfitting
- Explore advanced architectures (e.g., Transformers)

---

## 📖 Detailed Sections

### Section 1 — Data Preparation & Exploration

**Goal**: Collect, organize, and understand the dataset before model development.

**Tasks**:
- Download and organize the MS COCO dataset (images and captions)
- Resize all images to 299×299 pixels and normalize pixel values (0–1 range)
- Load and preview captions, analyze length distribution and vocabulary size

**Outcome**: Cleaned and standardized dataset, ready for feature extraction and caption processing.

---

### Section 2 — Caption Processing

**Goal**: Convert textual captions into numerical form suitable for training.

**Tasks**:
- Clean and tokenize captions using Keras Tokenizer
- Add special tokens `<start>` and `<end>` to mark caption boundaries
- Pad all caption sequences to a uniform length (max_len)

**Outcome**: Tokenizer file (`tokenizer.pkl`) and padded numerical caption sequences prepared for the model.

---

### Section 3 — Feature Extraction (CNN Encoder)

**Goal**: Extract visual features from images using a pre-trained CNN.

**Tasks**:
- Load InceptionV3 model pre-trained on ImageNet
- Remove its final classification layer to use it as a feature extractor
- Pass all images through the model to obtain feature vectors (8×8×2048)
- Save extracted features for reuse during training

**Outcome**: `features/` folder containing precomputed `.npy` feature files representing each image.

---

### Section 4 — Sequence Creation for Training

**Goal**: Prepare input-output pairs to teach the model to generate captions word by word.

**Tasks**:
- For each caption, create partial input sequences and their next-word targets
- Pad input sequences to a fixed max_len for uniformity
- Optionally convert datasets into a `tf.data.Dataset` format for efficient batching

**Outcome**: Model-ready arrays or TFRecords connecting image features to sequential text inputs.

---

### Section 5 — Model Architecture (Encoder-Decoder with Attention)

**Goal**: Build a deep learning architecture that combines visual and language understanding.

**Tasks**:
- Use the CNN encoder (from extracted features) to represent image content
- Implement the LSTM-based decoder to generate word sequences
- Add an attention mechanism to focus on relevant image regions during caption generation

**Outcome**: Fully compiled Encoder-Decoder model with Attention, with architecture summary saved as `model_summary.png`.

---

### Section 6 — Model Training

**Goal**: Train the encoder-decoder model to generate coherent captions.

**Tasks**:
- Split dataset into training (80%) and validation (20%) subsets
- Train using the Adam optimizer and categorical cross-entropy loss
- Monitor training with callbacks (model checkpointing, early stopping, LR scheduler)

**Outcome**: Trained model weights (`best_model.h5`), training logs, and loss/accuracy plots.

---

### Section 7 — Evaluation & Caption Generation

**Goal**: Evaluate the trained model's caption quality and generalization.

**Tasks**:
- Implement a `generate_caption()` function for inference
- Compute BLEU-1 to BLEU-4 scores using reference captions
- Display test images with their generated and actual captions for qualitative review

**Outcome**: Quantitative BLEU metrics and qualitative visual results demonstrating caption accuracy.

---

### Section 8 — Fine-Tuning & Enhancements

**Goal**: Improve model performance and robustness through optimization.

**Tasks**:
- Experiment with hyperparameters (learning rate, batch size, LSTM units)
- Add dropout layers or data augmentation to prevent overfitting
- Try advanced architectures (e.g., Transformer-based decoders) for comparison

**Outcome**: Optimized model with improved caption fluency and higher BLEU scores.

---

### Section 9 — Results, Analysis & Documentation

**Goal**: Present, interpret, and discuss project outcomes.

**Tasks**:
- Summarize training and evaluation results with visual plots and metrics
- Discuss success cases, limitations, and potential areas for improvement
- Document all findings, graphs, and tables within markdown cells for the report

**Outcome**: Clear, well-documented analysis section ready for inclusion in the final major project report.

---

### Section 10 — Saving, Export & Submission

**Goal**: Package the project for evaluation and future reuse.

**Tasks**:
- Save final artifacts — model (`best_model.h5`), tokenizer (`tokenizer.pkl`), generated captions, and metrics
- Export the entire notebook for submission
- Upload the project repository to GitHub for demonstration

**Outcome**: A complete, reproducible end-to-end project notebook aligned with the official project proposal.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pillow (PIL)
- tqdm
- nltk (for BLEU score calculation)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Image-Captioning-Team-C.git
   cd Image-Captioning-Team-C
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MS COCO dataset**:
   - Visit [MS COCO Dataset](https://cocodataset.org/#download)
   - Download images and annotations
   - Organize them in the project directory

5. **Download NLTK data** (for BLEU scores):
   ```python
   import nltk
   nltk.download('punkt')
   ```

---

## 🚀 Usage

### Training the Model

1. **Prepare the data**:
   ```python
   python data_preparation.py
   ```

2. **Extract features**:
   ```python
   python feature_extraction.py
   ```

3. **Train the model**:
   ```python
   python train_model.py
   ```

### Generating Captions

```python
from inference import generate_caption

# Load the trained model and tokenizer
caption = generate_caption('path/to/image.jpg')
print(caption)
```

### Evaluating the Model

```python
python evaluate_model.py
```

---

## 📁 Project Structure

```
Image-Captioning-Team-C/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
│   ├── images/              # MS COCO images
│   └── annotations/         # Caption annotations
│
├── features/                # Extracted image features
│   └── *.npy               # Feature files
│
├── models/                  # Saved models
│   ├── best_model.h5       # Trained model weights
│   └── tokenizer.pkl       # Tokenizer object
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_caption_processing.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_sequence_creation.ipynb
│   ├── 05_model_architecture.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_evaluation.ipynb
│   ├── 08_fine_tuning.ipynb
│   ├── 09_results_analysis.ipynb
│   └── 10_export_submission.ipynb
│
├── src/                     # Source code
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── model_architecture.py
│   ├── train_model.py
│   ├── inference.py
│   └── evaluation.py
│
├── results/                 # Output files
│   ├── plots/              # Training plots
│   ├── captions/           # Generated captions
│   └── metrics/            # Evaluation metrics
│
└── docs/                    # Additional documentation
    └── project_proposal.pdf
```

---

## 👥 Team Information

### Leadership
- **Team Lead**: Vijayabhaskar V
- **Co-Leads**: Adit Jain, Harshit Kumar

### Section-wise Allocation
- **Section 1**: [Team Member Name]
- **Section 2**: [Team Member Name]
- **Section 3**: [Team Member Name]
- **Section 4**: [Team Member Name]
- **Section 5**: [Team Member Name]
- **Section 6**: [Team Member Name]
- **Section 7**: [Team Member Name]
- **Section 8**: [Team Member Name]
- **Section 9**: [Team Member Name]
- **Section 10**: [Team Member Name]

> **Note**: Vijayabhaskar (Team Lead) & Adit Jain (Co-Lead) work collaboratively on each part, helping everyone to improve, modify, and optimize the code for the best overall project outcome.

---

## 📊 Results

### Model Performance

- **BLEU-1 Score**: [To be updated]
- **BLEU-2 Score**: [To be updated]
- **BLEU-3 Score**: [To be updated]
- **BLEU-4 Score**: [To be updated]

### Sample Captions

| Image | Generated Caption | Reference Caption |
|-------|------------------|-------------------|
| [Image 1] | "A cat is sitting on a mat" | "A cat sitting on a mat" |
| [Image 2] | "Two people are playing tennis" | "Two people playing tennis on a court" |

### Training Metrics

- **Training Loss**: [To be updated]
- **Validation Loss**: [To be updated]
- **Training Accuracy**: [To be updated]
- **Validation Accuracy**: [To be updated]

---

## 📚 References

1. **MS COCO Dataset**: [https://cocodataset.org/](https://cocodataset.org/)
2. **InceptionV3 Paper**: Szegedy, C., et al. "Rethinking the Inception Architecture for Computer Vision." CVPR 2016.
3. **Show, Attend and Tell**: Xu, K., et al. "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." ICML 2015.
4. **TensorFlow Documentation**: [https://www.tensorflow.org/](https://www.tensorflow.org/)
5. **Keras Documentation**: [https://keras.io/](https://keras.io/)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- MS COCO dataset creators and contributors
- TensorFlow and Keras development teams
- Open-source community for tools and libraries

---

## 📧 Contact

For questions or contributions, please contact:
- Adit Jain
- Vijayabhaskar V

---

**Last Updated**: [Date]

**Project Status**: 🚧 In Progress
