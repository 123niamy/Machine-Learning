# Generative AI

Projects exploring generative models and AI content creation.

## Projects

### GAN_iris_generator.py
**Generative Adversarial Network (GAN)** for synthetic Iris dataset generation.

**What it does:**
- Trains a Generator to create fake Iris flower measurements
- Trains a Discriminator to distinguish real from fake samples
- Uses adversarial training where both networks compete and improve

**Key Features:**
- PyTorch implementation with BatchNorm for stability
- BCELoss for proper gradient flow
- Visualization of training loss curves
- Real vs Fake sample comparison plots
- Generates synthetic data matching real Iris distribution

**Tech Stack:**
- PyTorch for neural networks
- scikit-learn for dataset and preprocessing
- matplotlib for visualization

**Usage:**
```bash
python GAN_iris_generator.py
```

**Output:**
- Training progress printed to console
- `gan_iris_results.png` - Loss curves and sample comparison visualization

---

## Generative AI Concepts Covered

- **GANs (Generative Adversarial Networks)**: Two-player game between Generator and Discriminator
- **Adversarial Training**: Networks compete to improve each other
- **Neural Network Architecture**: Custom Generator and Discriminator classes
- **Loss Functions**: Binary cross-entropy for classification
- **Batch Normalization**: Stabilizes training dynamics

---

**Coming Soon:**
- Text generation with transformers
- Image generation with diffusion models
- Prompt engineering for LLMs
