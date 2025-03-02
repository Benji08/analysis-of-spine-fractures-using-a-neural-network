import torch
import os
import nrrd
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from neural_network import Simple3DCNN

class Custom3DDataset(Dataset):
    def __init__(self, tensors, y1_labels, y2_labels):
        self.tensors = tensors
        self.y1_labels = y1_labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        x = self.tensors[idx].unsqueeze(0)  # dodajemy kanał
        y1 = self.y1_labels[idx]
        return x, y1


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

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.nrrd'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read the .nrrd file
                data, _ = nrrd.read(file_path)
                # Add the shape of the file to the set
                shapes.add(data.shape)
            except Exception as e:
                # Handle any errors during reading
                print(f"Error reading file {filename}: {e}")

    # Check if all shapes are identical
    if len(shapes) == 1:
        return True, shapes.pop()
    else:
        return False, shapes


def preprocess_and_convert(file_path, lower_bound=0, upper_bound=2000):
    """
    Preprocesses a single .nrrd file and converts it to a PyTorch tensor.

    Args:
        file_path (str): Path to the .nrrd file.
        lower_bound (float): Minimum value for clipping.
        upper_bound (float): Maximum value for clipping.
        standardize (bool): Whether to apply Z-score standardization.

    Returns:
        torch.Tensor: Preprocessed tensor.
    """
    # Step 1: Load the .nrrd file
    data, _ = nrrd.read(file_path)

    # Step 2: Clip the data to the specified range
    data = np.clip(data, lower_bound, upper_bound)

    #obecnie używam standaryzacji bez normalizacji
    # Step 3: Normalize the data to [0, 1]
    #data = (data - data.min()) / (data.max() - data.min())

    # Step 4: Optionally standardize the data
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    # Step 5: Convert to PyTorch tensor
    tensor = torch.tensor(data, dtype=torch.float32)

    # Extract y1 and y2 from filename
    filename = os.path.basename(file_path)
    if filename[7:9] == "--":
        y1 = 0
        y2 = 0
    else:
        y1 = 1
        code = filename[7:9]
        y2_mapping = {"A0": 1, "A1": 2, "A2": 3, "A3": 4, "A4": 5}
        y2 = y2_mapping.get(code, 0)  # domyślnie 0, jeśli kod nieznany
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
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Dodanie wymiarów batch i channel
    tensor_resized = F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)
    return tensor_resized.squeeze(0).squeeze(0)  # Usunięcie wymiarów batch i channel


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


def generate_augmented_tensors(tensor_list, num_tensors, augmentations):
    """
    Generates a specified number of augmented tensors by applying random augmentations
    to randomly chosen tensors from the input list.

    Args:
        tensor_list (list of torch.Tensor): List of input tensors.
        num_tensors (int): Number of augmented tensors to generate.
        augmentations (list): List of augmentation functions.

    Returns:
        list of torch.Tensor: List of newly augmented tensors.
    """
    augmented_tensors = []

    for _ in range(num_tensors):
        # Randomly select a tensor from the input list
        tensor = random.choice(tensor_list)

        # Randomly select an augmentation function
        augmentation = random.choice(augmentations)

        # Apply the augmentation to the tensor
        augmented_tensor = augmentation(tensor.clone())

        # Append the augmented tensor to the result list
        augmented_tensors.append(augmented_tensor)

    return augmented_tensors


# Ścieżka do folderu z danymi
folder_path = "C:/Users/barba/OneDrive/Pulpit/tomografia"
processed_tensors = []
Y1 = []
Y2 = []

# Process all .nrrd files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.nrrd'):
        file_path = os.path.join(folder_path, filename)
        tensor, y1, y2 = preprocess_and_convert(file_path, lower_bound=150, upper_bound=1500)
        tensor = resize_tensor(tensor)
        processed_tensors.append(tensor)
        Y1.append(y1)
        Y2.append(y2)
print(f"Generated {len(processed_tensors)} tensors. Size = {processed_tensors[0].size}")

# Define augmentation functions
augmentations = [
        flip_x,  # Flip along x-axis
        lambda t: translate(t, max_shift=10),  # Translate
        lambda t: add_gaussian_noise(t, mean=0, std=0.02),  # Add Gaussian noise
    ]
augmented_tensors = generate_augmented_tensors(processed_tensors, 5, augmentations)
print(f"Generated {len(augmented_tensors)} augmented tensors.")

all_tensors = processed_tensors + augmented_tensors
dataset = Custom3DDataset(all_tensors, Y1, Y2)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Inicjalizacja modelu, funkcji straty i optymalizatora
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Simple3DCNN().to(device)

criterion_y1 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Pętla treningowa
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_y1 = 0
    total = 0

    for inputs, labels_y1 in dataloader:
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

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy_y1 = correct_y1 / total

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy y1: {epoch_accuracy_y1:.4f}")
