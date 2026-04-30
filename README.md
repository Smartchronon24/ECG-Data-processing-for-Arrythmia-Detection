# 🫀 ECG Arrhythmia Detection using Deep Learning

## 📌 Overview

This project focuses on detecting and classifying cardiac arrhythmias from ECG (Electrocardiogram) signals using a hybrid deep learning approach. The system processes raw ECG signals, performs signal denoising and segmentation, and classifies heartbeats into clinically relevant categories using a CNN + BiLSTM architecture.

---

## 🚀 Key Features

* 📊 ECG signal preprocessing using wavelet denoising
* ❤️ Heartbeat segmentation around R-peaks
* 🔄 Reduction of 15 heartbeat classes into 5 major categories
* ⚖️ Handling class imbalance using undersampling + SMOTE
* 🧠 Deep learning model combining CNN, Gated Convolution, and BiLSTM
* 📈 Model evaluation using accuracy, confusion matrix, and advanced metrics

---

## 🧠 Problem Statement

ECG signals are often noisy and complex, making arrhythmia detection challenging. This project aims to:

* Clean and preprocess ECG signals
* Extract meaningful heartbeat segments
* Classify arrhythmias accurately using deep learning

---

## 📂 Dataset

The project uses the **MIT-BIH Arrhythmia Dataset**, which contains:

* ECG recordings
* Annotated heartbeat locations (R-peaks)
* Labels for different arrhythmia types

### Dataset Structure

```
mit-bih-arrhythmia-database-1.0.0/
├── 100.dat
├── 100.hea
├── 100.atr
├── 101.dat
├── 101.hea
├── 101.atr
...
```

---

## ⚙️ Workflow

### 1. Signal Denoising

* Wavelet transform (db5) used to remove noise from ECG signals

### 2. Heartbeat Segmentation

* Each heartbeat extracted as a window of 300 samples around R-peaks

### 3. Label Processing

* Original 15 classes reduced to 5 categories:

  * Normal
  * Supraventricular
  * Ventricular
  * Fusion
  * Unknown

### 4. Data Balancing

* Undersampling for majority class
* SMOTE for minority class augmentation

### 5. Model Training

* Hybrid architecture:

  * CNN layers for feature extraction
  * Gated Convolution for selective feature learning
  * BiLSTM for temporal dependencies

### 6. Evaluation

* Accuracy
* Confusion Matrix
* Sensitivity, Specificity, F1 Score

---

## 🧱 Model Architecture

* Conv1D + BatchNorm
* Gated Convolution Layer (custom)
* Pooling layers (Max + Avg)
* Bidirectional LSTM layers
* Fully Connected Dense layers
* Softmax output (5 classes)

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy & Pandas
* Scikit-learn
* Imbalanced-learn (SMOTE)
* PyWavelets
* WFDB
* Matplotlib & Seaborn

---

## ▶️ How to Run

### 1. Clone the Repository

```
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install Dependencies

```
pip install wfdb pywavelets seaborn tensorflow scikit-learn imbalanced-learn matplotlib
```

### 3. Add Dataset

Place the dataset folder in the project directory:

```
mit-bih-arrhythmia-database-1.0.0/
```

### 4. Run the Project

```
python mainfile.py
```

---

## 📊 Output

* Trained model saved as `.h5` file
* Accuracy and loss graphs
* Confusion matrix visualization
* Detailed classification report

---

## 📈 Results

The model achieves strong performance in classifying arrhythmia types by leveraging both spatial (CNN) and temporal (LSTM) features of ECG signals.

---

## 🔮 Future Improvements

* Real-time ECG monitoring integration
* Deployment as a mobile or web application
* Model optimization for faster inference
* Integration with wearable devices

---

## 🙌 Acknowledgements

* MIT-BIH Arrhythmia Dataset
* Open-source libraries and research community

---
