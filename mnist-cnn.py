import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Device Configuration (for M1)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Metal) device.")
elif torch.cuda.is_available(): # Fallback for other systems, though not M1
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

# 2. Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 3. MNIST Dataset and DataLoader
# Transformations: Convert to Tensor and Normalize
# MNIST mean and std dev (pre-calculated for the dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load training dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transform,
                                           download=True)

# Download and load test dataset
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transform)

# Data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

# 4. Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 1 channel (grayscale), Output: 32 channels, Kernel: 3x3, Padding: 1
        # (N, 1, 28, 28) -> (N, 32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # (N, 32, 28, 28) -> (N, 32, 14, 14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Input: 32 channels, Output: 64 channels, Kernel: 3x3, Padding: 1
        # (N, 32, 14, 14) -> (N, 64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # (N, 64, 14, 14) -> (N, 64, 7, 7)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output for the fully connected layer
        # 64 channels * 7x7 image size = 3136
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) # No softmax here, CrossEntropyLoss will apply it
        return x

model = SimpleCNN().to(device) # Move model to the selected device

# 5. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 6. Training Loop
print("Starting training...")
total_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct_train / total_train
    print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

print("Finished Training!")

# 7. Testing Loop
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for evaluation
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct_test / total_test
print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

# 8. (Optional) Visualize some predictions
def imshow(img, title):
    img = img.cpu().numpy() # move to cpu and convert to numpy
    # Unnormalize (reverse of transforms.Normalize((0.1307,), (0.3081,)))
    img = img * 0.3081 + 0.1307
    plt.imshow(np.transpose(img, (1, 2, 0)).squeeze(), cmap='gray') # .squeeze() if grayscale
    plt.title(title)
    plt.show()

# Get some random test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images_to_show = images[:8].to(device) # Show first 8 images from batch
labels_to_show = labels[:8].to(device)

outputs = model(images_to_show)
_, predicted = torch.max(outputs, 1)

print("\nSample Predictions:")
fig = plt.figure(figsize=(12, 6))
for idx in np.arange(8):
    ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
    # Use images[idx] directly as it's already on CPU or ready for imshow after .cpu()
    # imshow expects image in (C, H, W) or (H, W, C)
    img_display = images[idx].cpu() * 0.3081 + 0.1307 # Unnormalize
    ax.imshow(img_display.squeeze(), cmap='gray') # .squeeze() for single channel
    ax.set_title(f"Pred: {predicted[idx].item()}\nTrue: {labels[idx].item()}",
                 color=("green" if predicted[idx]==labels[idx] else "red"))
plt.tight_layout()
plt.show()
