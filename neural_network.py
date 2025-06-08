from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch


class Conv3DNet(nn.Module):
    def __init__(self, input_shape=(1, 32, 32, 16)):
        super(Conv3DNet, self).__init__()

        # --- Część konwolucyjna (Ekstraktor Cech) ---
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

        with torch.no_grad():
            _dummy_input = torch.randn(1, *input_shape)
            _dummy_output = self.features(_dummy_input)
            in_features_for_fc = _dummy_output.view(-1).shape[0]

        # --- Część klasyfikująca ---
        self.classifier = nn.Sequential(
            nn.Linear(in_features_for_fc, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Custom3DDataset(Dataset):
    def __init__(self, tensors, y1_labels):
        self.tensors = tensors
        self.y1_labels = y1_labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        x = self.tensors[idx]
        y1 = self.y1_labels[idx]
        return x, y1


class EarlyStopping:
    """
    Zatrzymuje trening, gdy monitorowana metryka przestaje się poprawiać.
    """

    def __init__(self, patience=7, mode="min", verbose=False, delta=0, path='best_model.pt', trace_func=print):
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        if self.mode == "min":
            self.val_score_min = np.inf
        else:
            self.val_score_min = -np.inf  # W trybie max zaczynamy od minus nieskończoności

    def __call__(self, score, model):
        if self.mode == "min":
            # Chcemy minimalizować (np. loss)
            if score < self.val_score_min - self.delta:
                self.save_checkpoint(score, model)
                self.val_score_min = score
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

        else:
            if score > self.val_score_min + self.delta:
                self.save_checkpoint(score, model)
                self.val_score_min = score
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.val_score_min:.6f} --> {score:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
