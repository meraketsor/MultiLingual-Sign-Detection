# Pose-Based Static Sign Language Recognition with DL for Turkish, Arabic, and American Sign Languages

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

## 📊 Abstract

This research presents a comprehensive comparative analysis of Vision Mamba (VMamba) against  other two deep learning models (Swin Transformer - ConvNeXt) for real-time sign language recognition across three distinct sign languages: Turkish (TSL), Arabic (ArSL), and American (ASL). Leveraging MediaPipe for robust hand landmark extraction, we evaluate 3 deep learning algorithms to establish benchmark performance metrics in cross-lingual gesture recognition.

## 🎯 Key Contributions

- **First comprehensive evaluation** of Vision Mamba for sign language recognition
- **Cross-lingual performance analysis** across TSL, ArSL, and ASL
- **Real-time implementation** with MediaPipe hand tracking
- **Comparative study** of 3 DL algorithms ( Vmamba - Swin Transformer - ConvNeXt)
- **Publicly available dataset** and implementation

## 🖼️ System Interface

![Sign Language Recognition Interface](images/interface.png)

*Real-time sign language recognition interface showing American Sign Language (ASL) detection*

## 📁 Download Resources

### Model Files
Due to large file sizes, download pre-trained models from:
**[Google Drive - Model Files](https://drive.google.com/drive/folders/141_wQe-TPp6tVFVTZmenO10VMpba2J1-?usp=drive_link)**

### Dataset Images
Download the complete dataset from:
**[Google Drive - Dataset Images](https://drive.google.com/drive/folders/162fJ1nPdBZfOexLC-RbfYPoecpGO1pSO?usp=drive_link)**

## 🧠 Algorithm Overview

### Vision Mamba (VMamba) - Primary Model
**Architecture**: State-space models with selective scanning mechanism  
**Advantages**: 
- Linear computational complexity
- Global receptive field
- Superior sequence modeling
- Enhanced cross-lingual generalization

### Comparative Models
1. **ConvNext** 
2. **Swin Transformer** 


## 📊 Dataset & Methodology

### Data Collection
- **Total Samples**: 130,500 annotated hand gestures
- **Languages**: TSL (43,500), ArSL (48,000), ASL (39,000)
- **Features**: 21 hand landmarks × 3 coordinates per hand
- **Validation**: 5-fold cross-validation

### Preprocessing Pipeline
1. MediaPipe Hand Landmark Extraction
2. Coordinate Normalization
3. Feature Vector Construction
4. Data Augmentation
5. Train-Test Split (80-20)





