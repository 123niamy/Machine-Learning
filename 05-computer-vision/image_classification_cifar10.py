# Image Classification - Complete ML Workflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Set working directory
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

print("🚀 Image Classification Workflow\n")

# ==================== STEP 1: DATA COLLECTION ====================
print("📊 Step 1: Data Collection")
# Using CIFAR-10 dataset (60,000 32x32 color images in 10 classes)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Download and load training data
train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                  download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                 download=True, transform=transform)

print(f"✅ Training images: {len(train_dataset)}")
print(f"✅ Test images: {len(test_dataset)}")

# ==================== STEP 2: DATA PREPARATION ====================
print("\n🔧 Step 2: Data Preparation")
# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
print(f"Classes: {classes}")

# ==================== STEP 3: DATA PREPROCESSING ====================
print("\n🧹 Step 3: Data Preprocessing")
# Already applied: ToTensor() and Normalization in transforms
print("✅ Images converted to tensors")
print("✅ Normalized to [-1, 1] range")

# ==================== STEP 4: DATA SPLITTING ====================
print("\n✂️ Step 4: Data Splitting")
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

print(f"✅ Training batches: {len(train_loader)}")
print(f"✅ Test batches: {len(test_loader)}")

# ==================== STEP 5: MODEL SELECTION ====================
print("\n🎯 Step 5: Model Selection - Convolutional Neural Network")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

model = SimpleCNN()
print("✅ CNN Model created")
print(f"   Architecture: 3 Conv layers + 3 FC layers")

# ==================== STEP 6: MODEL TRAINING ====================
print("\n🏋️ Step 6: Model Training")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 3  # Limited for quick demo
print(f"Training for {epochs} epochs...")

train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Print progress every 500 batches
        if (i + 1) % 500 == 0:
            print(f'   Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f'✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%')

print("\n✅ Training Complete!")

# ==================== STEP 7: PREDICTION ====================
print("\n🔮 Step 7: Making Predictions")
model.eval()

# Get a batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Make predictions
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

print(f"✅ Predictions made on {len(images)} test images")

# ==================== STEP 8: EVALUATION ====================
print("\n📈 Step 8: Model Evaluation")

correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Overall accuracy
accuracy = 100 * correct / total
print(f'\n🎯 Overall Test Accuracy: {accuracy:.2f}%')

# Per-class accuracy
print(f'\n📊 Per-Class Accuracy:')
for i in range(10):
    class_acc = 100 * class_correct[i] / class_total[i]
    print(f'   {classes[i]:8s}: {class_acc:.2f}%')

# ==================== VISUALIZATION ====================
print("\n📊 Creating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Training Loss Curve
axes[0, 0].plot(range(1, epochs+1), train_losses, 'b-', marker='o', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].grid(True, alpha=0.3)

# 2. Training Accuracy Curve
axes[0, 1].plot(range(1, epochs+1), train_accuracies, 'g-', marker='o', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Training Accuracy Over Time')
axes[0, 1].grid(True, alpha=0.3)

# 3. Sample Predictions
axes[1, 0].axis('off')
axes[1, 0].set_title('Sample Predictions')

# Show first 8 images with predictions
sample_images = images[:8]
sample_labels = labels[:8]
sample_preds = predicted[:8]

# Create grid: 2 rows x 4 columns of 32x32 images
grid_img = np.zeros((64, 128, 3))
for idx in range(8):
    img = sample_images[idx].numpy().transpose((1, 2, 0))
    img = (img * 0.5 + 0.5)  # Denormalize
    img = np.clip(img, 0, 1)  # Ensure valid range
    row = idx // 4
    col = idx % 4
    grid_img[row*32:(row+1)*32, col*32:(col+1)*32] = img

axes[1, 0].imshow(grid_img)
axes[1, 0].text(64, -5, 'Top: ' + ' | '.join([classes[sample_preds[i]] for i in range(4)]), 
                ha='center', fontsize=8)
axes[1, 0].text(64, 69, 'Bot: ' + ' | '.join([classes[sample_preds[i]] for i in range(4, 8)]), 
                ha='center', fontsize=8)

# 4. Per-Class Accuracy Bar Chart
class_accs = [100 * class_correct[i] / class_total[i] for i in range(10)]
axes[1, 1].bar(classes, class_accs, color='skyblue', edgecolor='navy')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Per-Class Accuracy')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('image_classification_results.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved as 'image_classification_results.png'")
plt.close()

print("\n" + "="*60)
print("🎉 Image Classification Workflow Complete!")
print("="*60)
