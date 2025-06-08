from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv3DNet(nn.Module):
    def __init__(self, input_shape=(1, 32, 32, 16)):
        super(Conv3DNet, self).__init__()

        # --- Część konwolucyjna (Ekstraktor Cech) ---
        # Używamy nn.Sequential dla czystości kodu
        self.features = nn.Sequential(
            # Blok 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),  # <--- NOWOŚĆ: Stabilizuje trening i ma efekt regularyzacji
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),  # <--- NOWOŚĆ
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Blok 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128), # <--- NOWOŚĆ
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256), # <--- NOWOŚĆ
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Blok 3
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512), # <--- NOWOŚĆ
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # --- Dynamiczne obliczanie rozmiaru dla warstwy FC ---
        # To jest NAJLEPSZA praktyka. Już nigdy nie będziesz musiał ręcznie zmieniać tej liczby!
        with torch.no_grad():
            _dummy_input = torch.randn(1, *input_shape)
            _dummy_output = self.features(_dummy_input)
            in_features_for_fc = _dummy_output.view(-1).shape[0]

        # --- Część klasyfikująca ---
        self.classifier = nn.Sequential(
            nn.Linear(in_features_for_fc, 1024),
            nn.ReLU(),
            nn.Dropout(0.5), # <--- NOWOŚĆ: Kluczowa warstwa do walki z overfittingiem!
            nn.Linear(1024, 1) # <--- POPRAWKA: JEDNO wyjście dla klasyfikacji binarnej z BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Spłaszczenie
        x = self.classifier(x)
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