# Computer Vision

Image classification, object detection, and visual recognition using deep learning.

## Projects

### image_classification_cifar10.py
**CNN Image Classification** - Complete ML workflow for multi-class image classification.

**What it does:**
- Classifies images into 10 categories (plane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- Trains a custom Convolutional Neural Network from scratch
- Achieves **71.43% accuracy** on CIFAR-10 dataset

**8-Step ML Workflow:**
1. **Data Collection** - Download CIFAR-10 (60,000 32x32 color images)
2. **Data Preparation** - Split into training (50,000) and test (10,000)
3. **Data Preprocessing** - Normalize images to [-1, 1] range
4. **Data Splitting** - Create batches for efficient training
5. **Model Selection** - Build custom CNN with 3 conv layers + 3 FC layers
6. **Model Training** - Train for 3 epochs with Adam optimizer
7. **Prediction** - Make predictions on test images
8. **Evaluation** - Calculate accuracy and per-class performance

**CNN Architecture:**
```
Input (3x32x32) 
  → Conv2D(32) + ReLU + MaxPool
  → Conv2D(64) + ReLU + MaxPool  
  → Conv2D(128) + ReLU + MaxPool
  → Flatten
  → FC(256) + Dropout
  → FC(128) + Dropout
  → FC(10) → Output
```

**Key Features:**
- PyTorch implementation
- Data augmentation with normalization
- Dropout for regularization
- CrossEntropyLoss for multi-class classification
- Batch training with DataLoader
- Comprehensive visualizations (loss curves, accuracy charts, sample predictions)

**Performance:**
- Overall Test Accuracy: **71.43%**
- Best performing classes:
  - Frog: 89.30%
  - Truck: 84.50%
  - Car: 83.30%
  - Plane: 82.40%

**Usage:**
```bash
python image_classification_cifar10.py
```

**Output:**
- Training progress printed to console
- `image_classification_results.png` - Comprehensive visualization dashboard
- Per-class accuracy breakdown

**Dataset:**
- CIFAR-10 (automatically downloaded)
- 60,000 images (32x32 RGB)
- 10 balanced classes
- 50,000 train / 10,000 test split

---

## Computer Vision Concepts Covered

### Convolutional Neural Networks (CNNs)
- **Convolutional Layers**: Automatic feature extraction from images
- **Pooling Layers**: Spatial dimension reduction
- **Activation Functions**: ReLU for non-linearity
- **Fully Connected Layers**: High-level reasoning and classification

### Training Techniques
- **Batch Processing**: Efficient GPU utilization
- **Dropout**: Prevent overfitting by randomly disabling neurons
- **Data Normalization**: Standardize input distributions
- **Loss Functions**: CrossEntropyLoss for multi-class problems
- **Optimization**: Adam optimizer with learning rate 0.001

### Evaluation Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **Per-Class Accuracy**: Performance breakdown by category
- **Training Curves**: Monitor loss and accuracy over epochs

---

## Tech Stack

- **PyTorch 2.9.1** - Deep learning framework
- **torchvision 0.24.1** - Computer vision utilities and datasets
- **OpenCV 4.12.0** - Image processing (available for future projects)
- **matplotlib** - Visualization
- **numpy** - Numerical operations

---

**Coming Soon:**
- Object detection with YOLO
- Image segmentation
- Transfer learning with pre-trained models (ResNet, VGG)
- Real-time video processing
- Face recognition
- Style transfer
