# Cybersecurity Attack Detection using Deep Learning

## Overview

This project uses a PyTorch neural network to detect suspicious cybersecurity events from system log data.

The model is trained on the BETH cybersecurity dataset and learns to classify events as either:

- `1` = Suspicious / Malicious Activity
- `0` = Benign Activity

The objective is to identify potentially malicious behavior from structured system event logs using supervised deep learning.

---

## Dataset

The dataset contains system event information including:

- processId
- threadId
- parentProcessId
- userId
- mountNamespace
- argsNum
- returnValue

Target variable:

- `sus_label`

Dataset sizes:


| Dataset    | Rows    |
| ---------- | ------- |
| Training   | 763,144 |
| Validation | 188,967 |
| Test       | 188,967 |


---

## Model Architecture

PyTorch Neural Network:

Input Layer → 16 Neurons → ReLU → Output Layer → Sigmoid

Training configuration:

- Optimizer: Adam
- Loss Function: Binary Cross Entropy (BCELoss)
- Epochs: 10
- Feature Scaling: StandardScaler

---

## Results


| Metric              | Score    |
| ------------------- | -------- |
| Validation Accuracy | 99.9952% |
| Test Accuracy       | 94.4773% |


---

## Technologies Used

- Python
- PyTorch
- Pandas
- Scikit-learn

---

## Installation

```bash
pip install -r requirements.txt

```

## Run

```bash
python main.py

```

---

## Output

The script:

- Loads cybersecurity log data
- Scales features
- Trains a neural network
- Evaluates validation and test accuracy
- Saves the trained model

Generated file:

```text
cyber_attack_model.pth

```

---

## Future Improvements

- Confusion Matrix Visualization
- Precision / Recall Analysis
- ROC-AUC Evaluation
- Hyperparameter Optimization
- Real-Time Threat Detection API
- Streamlit Dashboard

