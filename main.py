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
import matplotlib.pyplot as plt
from datetime import datetime


def verify_data_shape(folder_path):
    shapes = set()
    for filename in os.listdir(folder_path):
        if filename.endswith('.nrrd'):
            file_path = os.path.join(folder_path, filename)
            try:
                data, _ = nrrd.read(file_path)
                shapes.add(data.shape)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    if len(shapes) == 1:
        return True, shapes.pop()
    else:
        return False, shapes

def preprocess_and_convert(file_path, lower_bound=0, upper_bound=1200):
    data, _ = nrrd.read(file_path)
    data = np.clip(data, lower_bound, upper_bound)
    data = (data - data.min()) / (data.max() - data.min())
    tensor = torch.tensor(data, dtype=torch.float32)
    filename = os.path.basename(file_path)
    if filename[4:6] == "__" and filename[7:9] == "__":
        y1 = 0
        y2 = 0
    else:
        y1 = 1
        code = filename[7:9]
        y2_mapping = {"A0": 1, "A1": 2, "A2": 3, "A3": 4, "A4": 5}
        y2 = y2_mapping.get(code, 0)
    return tensor, y1, y2

def resize_tensor(tensor, target_size=(32, 32, 16)):
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor_resized = F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)
    return tensor_resized.squeeze(0).squeeze(0)

def flip_x(tensor):
    return tensor.flip(2)

def translate(tensor, max_shift=5):
    D, H, W = tensor.shape
    shift_h = random.randint(-max_shift, max_shift)
    shift_w = random.randint(-max_shift, max_shift)
    shift_d = random.randint(-max_shift, max_shift)
    tensor = F.pad(tensor, (max_shift, max_shift, max_shift, max_shift, max_shift, max_shift), mode='constant', value=0)
    tensor = tensor[max_shift + shift_d: max_shift + shift_d + D,
                    max_shift + shift_h: max_shift + shift_h + H,
                    max_shift + shift_w: max_shift + shift_w + W]
    return tensor

def add_gaussian_noise(tensor, mean=0, std=0.01):
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

def generate_augmented_tensors(tensors, y1, y2, num_tensors, augmentations):
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
    augmented_samples = torch.stack(augmented_samples)
    augmented_y1 = torch.tensor(augmented_y1)
    augmented_y2 = torch.tensor(augmented_y2)
    return augmented_samples, augmented_y1, augmented_y2

def print_class_distribution(labels, label_name):
    counter = Counter(labels.tolist())
    print(f"📊 Rozkład klas dla {label_name}:")
    for cls, count in sorted(counter.items()):
        print(f"  Klasa {cls}: {count} próbek")


folder_path = "C:/Users/bpawlak/Desktop/data_set"
processed_tensors = []
Y1 = []
Y2 = []

for root, dirs, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith('.nrrd'):
            file_path = os.path.join(root, filename)
            tensor, y1, y2 = preprocess_and_convert(file_path)
            tensor = resize_tensor(tensor)
            processed_tensors.append(tensor)
            Y1.append(y1)
            Y2.append(y2)

processed_tensors = torch.stack(processed_tensors)
Y1 = torch.tensor(Y1)
Y2 = torch.tensor(Y2)
print_class_distribution(Y1, "Y1 (binary)")

train_X, test_X, train_Y1, test_Y1, train_Y2, test_Y2 = train_test_split(
    processed_tensors, Y1, Y2, test_size=0.3, random_state=42
)
val_X, test_X, val_Y1, test_Y1, val_Y2, test_Y2 = train_test_split(
    test_X, test_Y1, test_Y2, test_size=0.5, random_state=42
)

augmentations = [flip_x, lambda t: translate(t, 10), lambda t: add_gaussian_noise(t, 0, 0.02)]
augmented_tensors, augmented_Y1, augmented_Y2 = generate_augmented_tensors(train_X, train_Y1, train_Y2, 600, augmentations)

train_X = torch.cat([train_X, augmented_tensors], dim=0)
train_Y1 = torch.cat([train_Y1, augmented_Y1], dim=0)
train_Y2 = torch.cat([train_Y2, augmented_Y2], dim=0)

print_class_distribution(train_Y1, "Y1 (binary) for training sample")

train_dataset = Custom3DDataset(train_X.unsqueeze(1), train_Y1)
val_dataset = Custom3DDataset(val_X.unsqueeze(1), val_Y1)
test_dataset = Custom3DDataset(test_X.unsqueeze(1), test_Y1)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(datetime.now())
model = Conv3DNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# --- Pętla treningowa ---
epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []



for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_y1 = 0
    total = 0

    for inputs, labels_y1 in train_loader:
        # --- POPRAWKA 1: ZMIANA KSZTAŁTU I TYPU ETYKIET ---
        labels_y1 = labels_y1.float().view(-1, 1)  # Must be (batch_size, 1) and float

        inputs, labels_y1 = inputs.to(device), labels_y1.to(device)

        optimizer.zero_grad()
        outputs_y1 = model(inputs)

        # Obliczenie straty jest już OK, bo model i etykiety mają ten sam kształt
        loss = criterion(outputs_y1, labels_y1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # --- POPRAWKA 2:POPRAWNA METODA OBLICZANIA PREDYKCJI DLA KLASYFIKACJI BINARNEJ ---
        # 1. Przepuść logity przez funkcję sigmoid, aby uzyskać prawdopodobieństwa [0, 1]
        probs = torch.sigmoid(outputs_y1)
        # 2. Ustaw próg 0.5, aby uzyskać finalną predykcję 0 lub 1
        predicted_y1 = (probs > 0.5).float()

        # Porównaj predykcje z etykietami
        correct_y1 += (predicted_y1 == labels_y1).sum().item()
        total += labels_y1.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy_y1 = correct_y1 / total

    # --- PĘTLA WALIDACYJNA (te same poprawki) ---
    model.eval()
    val_loss = 0.0
    correct_y1_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels_y1 in val_loader:
            # --- POPRAWKA 1 (walidacja) ---
            labels_y1 = labels_y1.float().view(-1, 1)

            inputs, labels_y1 = inputs.to(device), labels_y1.to(device)
            outputs_y1 = model(inputs)
            loss_y1 = criterion(outputs_y1, labels_y1)
            val_loss += loss_y1.item()

            # --- POPRAWKA 2 (walidacja) ---
            probs = torch.sigmoid(outputs_y1)
            predicted_y1 = (probs > 0.5).float()

            correct_y1_val += (predicted_y1 == labels_y1).sum().item()
            total_val += labels_y1.size(0)

    val_loss /= len(val_loader)
    val_accuracy_y1 = correct_y1_val / total_val

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    train_accuracies.append(epoch_accuracy_y1)
    val_accuracies.append(val_accuracy_y1)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy_y1:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy_y1:.4f}")

# --- Wykresy metryk ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Val Loss", color='orange')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy", color='green')
plt.plot(val_accuracies, label="Val Accuracy", color='red')
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# --- Ewaluacja na zbiorze testowym ---
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels_y1 in test_loader:
        # --- POPRAWKA 1 (test) ---
        labels_y1 = labels_y1.float().view(-1, 1)

        inputs, labels_y1 = inputs.to(device), labels_y1.to(device)
        outputs_y1 = model(inputs)

        # --- POPRAWKA 2 (test) ---
        probs = torch.sigmoid(outputs_y1)
        predicted_y1 = (probs > 0.5).float()

        all_preds.extend(predicted_y1.cpu().numpy())
        all_labels.extend(labels_y1.cpu().numpy())

# Reszta kodu ewaluacji jest OK
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
# Dodaj zero_division=0, aby uniknąć warningu, gdyby model nie przewidział jakiejś klasy
class_report = classification_report(all_labels, all_preds, zero_division=0)

print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(datetime.now())