from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class Conv3DNet(nn.Module):
    def __init__(self, num_classes=2):
        super(Conv3DNet, self).__init__()

        # Blok 1 - Warstwy konwolucyjne 3D
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)  # Z 1 kanału wejściowego do 32 wyjściowych
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Pooling 2x2x2

        # Blok 2 - Kolejne warstwy konwolucyjne
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Blok 3 - Warstwy konwolucyjne
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # Pooling 2x2x2

        # Warstwa w pełni połączona
        self.fc1 = nn.Linear(512 * 4 * 4 * 4*4, 1024)  # Przyjmujemy, że rozmiar po warstwach poolingowych to 4x4x4
        self.fc2 = nn.Linear(1024, num_classes)  # Liczba klas na wyjściu (np. 2 dla klasyfikacji binarnej)

    def forward(self, x):
        # Blok 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # Blok 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # Blok 3
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        # Przekształcenie do wektora (flatten)
        x = x.view(x.size(0), -1)

        # W pełni połączona sieć
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class Custom3DDataset(Dataset):
    def __init__(self, tensors, y1_labels):
        self.tensors = tensors
        self.y1_labels = y1_labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        x = self.tensors[idx]  # dodajemy kanał
        y1 = self.y1_labels[idx]
        return x, y1