import torch
import os
import nrrd
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from neural_network import Conv3DNet, Custom3DDataset, EarlyStopping
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime


def preprocess_and_convert(file_path, lower_bound=0, upper_bound=1200):
    """
    Reads an NRRD file, preprocesses it, and extracts a binary label from the filename.

    Preprocessing includes:
    1. Clipping voxel values to a specified range (Hounsfield units).
    2. Normalizing the clipped data to a [0, 1] range.
    3. Converting the NumPy array to a PyTorch tensor.

    The label is determined based on a specific pattern in the filename.

    Args:
        file_path (str): The path to the .nrrd file.
        lower_bound (int): The lower Hounsfield unit bound for clipping.
        upper_bound (int): The upper Hounsfield unit bound for clipping.

    Returns:
        tuple[torch.Tensor, int]: A tuple containing the preprocessed 3D tensor
                                   and its binary label (0 or 1).
    """
    data, _ = nrrd.read(file_path)
    data = np.clip(data, lower_bound, upper_bound)
    data = (data - data.min()) / (data.max() - data.min())
    tensor = torch.tensor(data, dtype=torch.float32)
    filename = os.path.basename(file_path)
    if filename[4:6] == "__" and filename[7:9] == "__":
        y1 = 0
    else:
        y1 = 1
    return tensor, y1

def resize_tensor(tensor, target_size=(32, 32, 16)):
    """
    Resizes a 3D tensor to a target size using trilinear interpolation.

    Args:
        tensor (torch.Tensor): The input 3D tensor (D, H, W).
        target_size (tuple[int, int, int]): The target size (D, H, W).

    Returns:
        torch.Tensor: The resized 3D tensor.
    """
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor_resized = F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)
    return tensor_resized.squeeze(0).squeeze(0)

def flip_x(tensor):
    """
    Augmentation function: Flips a tensor along the 'width' axis (dimension 2).

    Args:
        tensor (torch.Tensor): The input 3D tensor.

    Returns:
        torch.Tensor: The flipped tensor.
    """
    return tensor.flip(2)

def translate(tensor, max_shift=5):
    """
    Augmentation function: Randomly translates a tensor along all three dimensions.

    Args:
        tensor (torch.Tensor): The input 3D tensor.
        max_shift (int): The maximum shift amount in any direction.

    Returns:
        torch.Tensor: The translated tensor.
    """
    D, H, W = tensor.shape
    shift_d = random.randint(-max_shift, max_shift)
    shift_h = random.randint(-max_shift, max_shift)
    shift_w = random.randint(-max_shift, max_shift)
    tensor = torch.roll(tensor, shifts=(shift_d, shift_h, shift_w), dims=(0, 1, 2))
    return tensor

def add_gaussian_noise(tensor, mean=0, std=0.01):
    """
    Augmentation function: Adds Gaussian noise to the tensor.

    Args:
        tensor (torch.Tensor): The input 3D tensor.
        mean (float): The mean of the Gaussian noise.
        std (float): The standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: The tensor with added noise.
    """
    return tensor + torch.randn_like(tensor) * std + mean

def generate_augmented_tensors(tensors, y1, num_tensors_to_add, augmentations):
    """
    Generates new tensor samples by applying augmentations to the minority class.

    This function is used for oversampling to balance the dataset.

    Args:
        tensors (torch.Tensor): The original dataset tensors.
        y1 (torch.Tensor): The labels for the original tensors.
        num_tensors_to_add (int): The number of new samples to generate.
        augmentations (list[callable]): A list of augmentation functions to apply.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the augmented tensors
                                           and their corresponding labels.
    """
    counter = Counter(y1.tolist())
    minority_class = min(counter, key=counter.get)
    minority_indices = (y1 == minority_class).nonzero(as_tuple=True)[0]

    if len(minority_indices) == 0:
        print("Warning: No samples of the minority class found to augment.")
        return torch.tensor([]), torch.tensor([])

    augmented_samples = []
    augmented_y1 = []
    for _ in range(num_tensors_to_add):
        idx = random.choice(minority_indices)
        tensor = tensors[idx]
        augmentation = random.choice(augmentations)
        augmented_tensor = augmentation(tensor.clone())
        augmented_samples.append(augmented_tensor)
        augmented_y1.append(minority_class)

    return torch.stack(augmented_samples), torch.tensor(augmented_y1)

def print_class_distribution(labels, label_name):
    """
    Prints the class distribution for a given set of labels.

    Args:
        labels (torch.Tensor or list): A tensor or list of labels.
        label_name (str): A descriptive name for the dataset being analyzed.
    """
    counter = Counter(labels.tolist())
    print(f"📊 Class distribution for {label_name}:")
    for cls, count in sorted(counter.items()):
        print(f"  Class {cls}: {count} samples")


if __name__ == '__main__':
    # Define hyperparameters and constants for the experiment.
    FOLDER_PATH = "C:/Users/bpawlak/Desktop/data_set"
    TARGET_SIZE = (32, 32, 16)
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # Data Loading and Preprocessing
    print("Loading and preprocessing data...")
    processed_tensors, Y1, Y2 = [], [], []
    for root, _, files in os.walk(FOLDER_PATH):
        for filename in files:
            if filename.endswith('.nrrd'):
                file_path = os.path.join(root, filename)
                tensor, y1 = preprocess_and_convert(file_path)
                tensor = resize_tensor(tensor, TARGET_SIZE)
                processed_tensors.append(tensor)
                Y1.append(y1)

    processed_tensors = torch.stack(processed_tensors)
    Y1 = torch.tensor(Y1)
    print_class_distribution(Y1, "Y1 - Full Dataset")

    # Data Splitting
    train_val_X, test_X, train_val_Y1, test_Y1 = train_test_split(
        processed_tensors, Y1, test_size=0.2, random_state=42, stratify=Y1)
    train_X, val_X, train_Y1, val_Y1 = train_test_split(
        train_val_X, train_val_Y1, test_size=0.25, random_state=42, stratify=train_val_Y1)

    # Augmentation (Balancing)
    print("Performing augmentation to balance the training set...")
    num_majority = (train_Y1 == 0).sum().item()
    num_minority = (train_Y1 == 1).sum().item()
    num_to_add = num_majority - num_minority

    if num_to_add > 0:
        augmentations = [flip_x, lambda t: translate(t, 2), lambda t: add_gaussian_noise(t, 0, 0.01)]
        augmented_tensors, augmented_Y1 = generate_augmented_tensors(train_X, train_Y1, num_to_add, augmentations)
        train_X = torch.cat([train_X, augmented_tensors], dim=0)
        train_Y1 = torch.cat([train_Y1, augmented_Y1], dim=0)
    print_class_distribution(train_Y1, "Y1 - Training set after augmentation")

    # Dataset and DataLoader Creation
    train_dataset = Custom3DDataset(train_X.unsqueeze(1), train_Y1)
    val_dataset = Custom3DDataset(val_X.unsqueeze(1), val_Y1)
    test_dataset = Custom3DDataset(test_X.unsqueeze(1), test_Y1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Loss, and Optimizer Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Conv3DNet(input_shape=(1, *TARGET_SIZE)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max", verbose=True, path='best_model_offline_aug.pt')

    # Training and Validation Loop
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels_y1 in train_loader:
            labels_y1 = labels_y1.float().view(-1, 1)
            inputs, labels_y1 = inputs.to(device), labels_y1.to(device)

            # Forward pass, loss calculation, and backpropagation
            optimizer.zero_grad()
            outputs_y1 = model(inputs)
            loss = criterion(outputs_y1, labels_y1)
            loss.backward()
            optimizer.step()

            # Accumulate metrics for the epoch
            running_loss += loss.item()
            probs = torch.sigmoid(outputs_y1)
            predicted = (probs > 0.5).float()
            total += labels_y1.size(0)
            correct += (predicted == labels_y1).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels_y1 in val_loader:
                labels_y1_shaped = labels_y1.float().view(-1, 1)
                inputs, labels_y1_gpu = inputs.to(device), labels_y1_shaped.to(device)

                # Calculate validation metrics for the epoch
                outputs_y1 = model(inputs)
                loss_y1 = criterion(outputs_y1, labels_y1_gpu)
                val_loss += loss_y1.item()

                probs = torch.sigmoid(outputs_y1)
                predicted = (probs > 0.5).float()

                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels_y1_shaped.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)


        val_epoch_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        # Log epoch results
        print(f"Epoch {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}, "
              f"Val Macro F1: {val_macro_f1:.4f}")

        # Early Stopping Check
        early_stopping(val_macro_f1, model)
        if early_stopping.early_stop:
            print("Early stopping based on Val Macro F1 score.")
            break

    # Final Evaluation on Test Set
    print("\nLoading the best model for final evaluation...")
    model.load_state_dict(torch.load('best_model_offline_aug.pt'))

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels_y1 in test_loader:
            labels_y1 = labels_y1.float().view(-1, 1)
            inputs, labels_y1 = inputs.to(device), labels_y1.to(device)
            outputs_y1 = model(inputs)

            probs = torch.sigmoid(outputs_y1)
            predicted = (probs > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_y1.cpu().numpy())

    # Calculate and display final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, zero_division=0)

    print(f"\n✅ Final evaluation on the test set at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()