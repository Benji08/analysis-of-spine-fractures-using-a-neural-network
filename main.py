import torch
import os
import nrrd
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from neural_network import Conv3DNet, Custom3DDataset
from collections import Counter


def verify_data_shape(folder_path):
    """
    Check if all .nrrd files in the folder have the same shape.

    Args:
        folder_path (str): Path to the folder containing .nrrd files.

    Returns:
        tuple: (bool, set) - True and the common shape if all files have the same shape,
               False and a set of unique shapes otherwise.
    """
    shapes = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.nrrd'):
            file_path = os.path.join(folder_path, filename)
            try:
                data, _ = nrrd.read(file_path)
                shapes.add(data.shape)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # Check if all shapes are identical
    if len(shapes) == 1:
        return True, shapes.pop()
    else:
        return False, shapes

def preprocess_and_convert(file_path, lower_bound=0, upper_bound=1200):
    """
    Preprocesses a single .nrrd file and converts it to a PyTorch tensor.

    Args:
        file_path (str): Path to the .nrrd file.
        lower_bound (float): Minimum value for clipping.
        upper_bound (float): Maximum value for clipping.

    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    # Step 1: Load the .nrrd file
    data, _ = nrrd.read(file_path)

    # Step 2: Clip the data to the specified range
    data = np.clip(data, lower_bound, upper_bound)

    # Currently using standardization without normalization
    # Step 3: Normalize the data to [0, 1]
    data = (data - data.min()) / (data.max() - data.min())

    # Currently using normalization without standardization
    # Step 4: Standardize the data
    # mean = np.mean(data)
    # std = np.std(data)
    # data = (data - mean) / std

    # Step 5: Convert to PyTorch tensor
    tensor = torch.tensor(data, dtype=torch.float32)

    # Extract y1 and y2 from filename
    filename = os.path.basename(file_path)
    if filename[7:9] == "--" or filename[7:9] == "A0" or filename[7:9] == "A1":
        y1 = 0
        y2 = 0
    else:
        y1 = 1
        code = filename[7:9]
        y2_mapping = {"A0": 1, "A1": 2, "A2": 3, "A3": 4, "A4": 5}
        y2 = y2_mapping.get(code, 0)  # Default to 0 if the code is unknown
    return tensor, y1, y2

def resize_tensor(tensor, target_size=(64, 64, 32)):
    """
    Resizes a 3D tensor to a fixed size using trilinear interpolation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (D, H, W).
        target_size (tuple): Desired output shape (D, H, W).

    Returns:
        torch.Tensor: Resized tensor.
    """
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    tensor_resized = F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)
    return tensor_resized.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def flip_x(tensor):
    """
    Flips a 3D tensor along the x-axis (width).
    Args:
        tensor (torch.Tensor): Input 3D tensor of shape (D, H, W).
    Returns:
        torch.Tensor: Flipped tensor.
    """
    tensor = tensor.flip(2)
    return tensor

def translate(tensor, max_shift=5):
    """
    Translates a 3D tensor within a specified range along all axes.
    Args:
        tensor (torch.Tensor): Input 3D tensor of shape (D, H, W).
        max_shift (int): Maximum number of pixels to shift along each axis.
    Returns:
        torch.Tensor: Translated tensor.
    """
    D, H, W = tensor.shape
    shift_h = random.randint(-max_shift, max_shift)
    shift_w = random.randint(-max_shift, max_shift)
    shift_d = random.randint(-max_shift, max_shift)

    # Pad and crop to simulate translation
    tensor = F.pad(tensor, (max_shift, max_shift, max_shift, max_shift, max_shift, max_shift), mode='constant', value=0)
    tensor = tensor[max_shift + shift_d: max_shift + shift_d + D,
                    max_shift + shift_h: max_shift + shift_h + H,
                    max_shift + shift_w: max_shift + shift_w + W]
    return tensor

def add_gaussian_noise(tensor, mean=0, std=0.01):
    """
    Adds Gaussian noise to a 3D tensor.
    Args:
        tensor (torch.Tensor): Input 3D tensor.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    Returns:
        torch.Tensor: Tensor with added noise.
    """
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

def generate_augmented_tensors(tensors, y1, y2, num_tensors, augmentations):
    """
    Generates additional sets of tensors by augmenting existing tensors.

    Args:
        tensors (torch.Tensor): Input tensor with shape (N, D, H, W).
        y1 (torch.Tensor): Labels y1 with shape (N,).
        y2 (torch.Tensor): Labels y2 with shape (N,).
        num_tensors (int): Number of new tensors to generate.
        augmentations (list): List of augmentation functions.

    Returns:
        torch.Tensor: New tensor samples with shape (num_tensors, D, H, W).
        torch.Tensor: New labels y1 with shape (num_tensors,).
        torch.Tensor: New labels y2 with shape (num_tensors,).
    """
    counter = Counter(y1.tolist())
    minority_class = min(counter, key=counter.get)
    minority_indices = (y1 == minority_class).nonzero(as_tuple=True)[0]

    N, D, H, W = tensors.shape
    augmented_samples = []
    augmented_y1 = []
    augmented_y2 = []

    for _ in range(num_tensors):
        idx = random.choice(minority_indices)
        tensor = tensors[idx]
        label_y1 = y1[idx]
        label_y2 = y2[idx]

        augmentation = random.choice(augmentations)
        augmented_tensor = augmentation(tensor.clone())

        augmented_samples.append(augmented_tensor)
        augmented_y1.append(label_y1)
        augmented_y2.append(label_y2)

    augmented_samples = torch.stack(augmented_samples)  # (num_tensors, D, H, W)
    augmented_y1 = torch.tensor(augmented_y1)
    augmented_y2 = torch.tensor(augmented_y2)

    return augmented_samples, augmented_y1, augmented_y2

def print_class_distribution(labels, label_name):
    label_list = labels.tolist()
    counter = Counter(label_list)
    print(f"📊 Rozkład klas dla {label_name}:")
    for cls, count in sorted(counter.items()):
        print(f"  Klasa {cls}: {count} próbek")


# Path to data folder
folder_path = "C:/Users/barba/OneDrive/Pulpit/tomografia"
processed_tensors = []
Y1 = []
Y2 = []

# Process all .nrrd files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.nrrd'):
        file_path = os.path.join(folder_path, filename)
        tensor, y1, y2 = preprocess_and_convert(file_path)
        tensor = resize_tensor(tensor)
        processed_tensors.append(tensor)
        Y1.append(y1)
        Y2.append(y2)

processed_tensors = torch.stack(processed_tensors)  # (N, D, H, W)
Y1 = torch.tensor(Y1)
Y2 = torch.tensor(Y2)
print_class_distribution(Y1, "Y1 (binary)")

print(f"Generated {len(processed_tensors)} tensors. Shape = {processed_tensors[0].shape}")
# Division into sets: 70% train, 15% val, 15% test
train_X, test_X, train_Y1, test_Y1, train_Y2, test_Y2 = train_test_split(
    processed_tensors, Y1, Y2, test_size=0.3, random_state=42
)

val_X, test_X, val_Y1, test_Y1, val_Y2, test_Y2 = train_test_split(
    test_X, test_Y1, test_Y2, test_size=0.5, random_state=42
)

# Define augmentation functions
augmentations = [
        flip_x,  # Flip along x-axis
        lambda t: translate(t, max_shift=10),  # Translate
        lambda t: add_gaussian_noise(t, mean=0, std=0.02),  # Add Gaussian noise
    ]
augmented_tensors, augmented_Y1, augmented_Y2 = generate_augmented_tensors(train_X, train_Y1, train_Y2, 30, augmentations)
print(f"Generated {len(augmented_tensors)} augmented tensors.")
print_class_distribution(train_Y1, "Y1 (binary) for training sample")

# Combining original and augmented training data
train_X = torch.cat([train_X] + [augmented_tensors], dim=0)
train_Y1 = torch.cat([train_Y1] + [augmented_Y1], dim=0)
train_Y2 = torch.cat([train_Y2] + [augmented_Y2], dim=0)
print_class_distribution(train_Y1, "Y1 (binary) for training sample")


# Creation of DataLoaders
train_dataset = Custom3DDataset(train_X.unsqueeze(1), train_Y1)  # (N, 1, D, H, W)
val_dataset = Custom3DDataset(val_X.unsqueeze(1), val_Y1)
test_dataset = Custom3DDataset(test_X.unsqueeze(1), test_Y1)
print(f"📊 Liczba próbek w zbiorze treningowym: {len(train_dataset)}")
print(f"📊 Liczba próbek w zbiorze walidacyjnym: {len(val_dataset)}")
print(f"📊 Liczba próbek w zbiorze testowym: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialization of the model, loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = Conv3DNet().to(device)

criterion_y1 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_y1 = 0
    total = 0

    for inputs, labels_y1 in train_loader:
        inputs, labels_y1 = inputs.to(device), labels_y1.to(device)
        optimizer.zero_grad()
        outputs_y1 = model(inputs)
        loss_y1 = criterion_y1(outputs_y1, labels_y1)
        loss = loss_y1
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted_y1 = torch.max(outputs_y1, 1)
        correct_y1 += (predicted_y1 == labels_y1).sum().item()
        total += labels_y1.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy_y1 = correct_y1 / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct_y1_val = 0
    total_val = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels_y1 in val_loader:
            inputs, labels_y1 = inputs.to(device), labels_y1.to(device)

            outputs_y1 = model(inputs)
            loss_y1 = criterion_y1(outputs_y1, labels_y1)

            val_loss += loss_y1.item()
            _, predicted_y1 = torch.max(outputs_y1, 1)
            correct_y1_val += (predicted_y1 == labels_y1).sum().item()
            total_val += labels_y1.size(0)

            all_preds.extend(predicted_y1.cpu().numpy())  # przesuń do CPU i dodaj do listy
            all_labels.extend(labels_y1.cpu().numpy())

            # Wypisz predykcje na bieżąco
            for i in range(len(predicted_y1)):
                print(f"True label: {labels_y1[i].item()}, Predicted label: {predicted_y1[i].item()}")

    val_loss /= len(val_loader)
    val_accuracy_y1 = correct_y1_val / total_val

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy_y1:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy_y1:.4f}")


# Testing the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels_y1 in test_loader:
        inputs, labels_y1 = inputs.to(device), labels_y1.to(device)

        outputs_y1 = model(inputs)
        _, predicted_y1 = torch.max(outputs_y1, 1)

        all_preds.extend(predicted_y1.cpu().numpy())
        all_labels.extend(labels_y1.cpu().numpy())

# Metrics calculation
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds)

print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
