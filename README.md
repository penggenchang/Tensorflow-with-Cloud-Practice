# TensorFlow with Cloud Practice 🧠☁️  
CIFAR-10 Convolutional Neural Network (CNN) — trained on both **AWS EC2** and **Azure Virtual Machines** using TensorFlow.

---

## 📌 Project Overview

This project demonstrates how to:
- Build and train a CNN for image classification on the CIFAR-10 dataset
- Run training on **cloud CPU/GPU environments** (AWS EC2, Azure VM)
- Track metrics using both **CSVLogger** and **TensorBoard**
- Save trained models in `.h5` format

---

## 🏗️ Project Structure
pythonProject/ ├── models/ │ └── cnn_model.py ├── data/ │ └── loader.py ├── training/ │ └── train.py ├── logs/ │ ├── training_log.csv │ └── TensorBoard event files ├── cnn_cifar10_model.h5 └── requirements.txt
## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/penggenchang/Tensorflow-with-Cloud-Practice.git
cd Tensorflow-with-Cloud-Practice```

###2. Set up environment

```bash
git clone https://github.com/penggenchang/Tensorflow-with-Cloud-Practice.git
cd Tensorflow-with-Cloud-Practice
