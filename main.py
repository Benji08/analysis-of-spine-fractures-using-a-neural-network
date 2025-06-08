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
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor_resized = F.interpolate(tensor, size=target_size, mode='trilinear', align_corners=False)
    return tensor_resized.squeeze(0).squeeze(0)

def flip_x(tensor):
    return tensor.flip(2)

def translate(tensor, max_shift=5):
    D, H, W = tensor.shape
    shift_d = random.randint(-max_shift, max_shift)
    shift_h = random.randint(-max_shift, max_shift)
    shift_w = random.randint(-max_shift, max_shift)
    tensor = torch.roll(tensor, shifts=(shift_d, shift_h, shift_w), dims=(0, 1, 2))
    return tensor

def add_gaussian_noise(tensor, mean=0, std=0.01):
    return tensor + torch.randn_like(tensor) * std + mean

def generate_augmented_tensors(tensors, y1, num_tensors_to_add, augmentations):
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
    counter = Counter(labels.tolist())
    print(f"📊 Rozkład klas dla {label_name}:")
    for cls, count in sorted(counter.items()):
        print(f"  Klasa {cls}: {count} próbek")


if __name__ == '__main__':
    # --- Konfiguracja ---
    FOLDER_PATH = "C:/Users/bpawlak/Desktop/data_set"
    TARGET_SIZE = (32, 32, 16)
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15

    # --- Wczytywanie i Przetwarzanie Danych ---
    print("Wczytywanie i przetwarzanie danych...")
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
    print_class_distribution(Y1, "Y1 - Cały zbiór")

    # --- Podział na Zbiory Treningowe, Walidacyjne i Testowe ---
    train_val_X, test_X, train_val_Y1, test_Y1 = train_test_split(
        processed_tensors, Y1, test_size=0.2, random_state=42, stratify=Y1)
    train_X, val_X, train_Y1, val_Y1 = train_test_split(
        train_val_X, train_val_Y1, test_size=0.25, random_state=42, stratify=train_val_Y1)  # 0.25 * 0.8 = 0.2

    # --- Augmentacja "Offline" ---
    print("Przeprowadzanie augmentacji 'offline'...")
    num_majority = (train_Y1 == 0).sum().item()
    num_minority = (train_Y1 == 1).sum().item()
    num_to_add = num_majority - num_minority

    if num_to_add > 0:
        augmentations = [flip_x, lambda t: translate(t, 2), lambda t: add_gaussian_noise(t, 0, 0.01)]
        augmented_tensors, augmented_Y1 = generate_augmented_tensors(train_X, train_Y1, num_to_add, augmentations)
        train_X = torch.cat([train_X, augmented_tensors], dim=0)
        train_Y1 = torch.cat([train_Y1, augmented_Y1], dim=0)

    print_class_distribution(train_Y1, "Y1 - Zbiór treningowy po augmentacji")

    # --- Tworzenie Datasetów i DataLoaderów ---
    train_dataset = Custom3DDataset(train_X.unsqueeze(1), train_Y1)
    val_dataset = Custom3DDataset(val_X.unsqueeze(1), val_Y1)
    test_dataset = Custom3DDataset(test_X.unsqueeze(1), test_Y1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Inicjalizacja Modelu, Straty, Optymalizatora ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Używane urządzenie: {device}")

    model = Conv3DNet(input_shape=(1, *TARGET_SIZE)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max", verbose=True, path='best_model_offline_aug.pt')

    # --- Pętla Treningowa ---
    print(f"\nRozpoczynanie treningu o {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels_y1 in train_loader:
            labels_y1 = labels_y1.float().view(-1, 1)
            inputs, labels_y1 = inputs.to(device), labels_y1.to(device)

            optimizer.zero_grad()
            outputs_y1 = model(inputs)
            loss = criterion(outputs_y1, labels_y1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs_y1)
            predicted = (probs > 0.5).float()

            total += labels_y1.size(0)
            correct += (predicted == labels_y1).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Pętla Walidacyjna
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():
            for inputs, labels_y1 in val_loader:
                labels_y1_shaped = labels_y1.float().view(-1, 1)  # Użyj nowej zmiennej
                inputs, labels_y1_gpu = inputs.to(device), labels_y1_shaped.to(device)

                outputs_y1 = model(inputs)
                loss_y1 = criterion(outputs_y1, labels_y1_gpu)
                val_loss += loss_y1.item()

                probs = torch.sigmoid(outputs_y1)
                predicted = (probs > 0.5).float()

                # Zbieramy wszystkie predykcje i etykiety z całej epoki walidacyjnej
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels_y1_shaped.cpu().numpy())

        val_epoch_loss = val_loss / len(val_loader)

        # Obliczamy metryki dla całej epoki walidacyjnej

        val_epoch_accuracy = accuracy_score(all_val_labels, all_val_preds)
        # Obliczamy Macro F1-score, który jest naszą nową metryką do śledzenia
        val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        # Aktualizujemy listy do wykresów
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        # Printuj obie metryki, aby mieć pełen obraz
        print(f"Epoch {epoch + 1}/{EPOCHS}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.4f}, "
              f"Val Macro F1: {val_macro_f1:.4f}")  # <--- NOWA METRYKA W LOGACH

        # Przekazujemy do EarlyStopping naszą nową, mądrzejszą metrykę
        early_stopping(val_macro_f1, model)
        if early_stopping.early_stop:
            print("Early stopping based on Val Macro F1 score.")
            break

    # --- Ewaluacja na Zbiorze Testowym z Najlepszym Modelem ---
    print("\nŁadowanie najlepszego modelu do ewaluacji...")
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

    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, zero_division=0)

    print(f"\n✅ Końcowa ewaluacja na zbiorze testowym o {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # --- Wykresy Metryk ---
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