from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch


class Conv3DNet(nn.Module):
    """
    A 3D Convolutional Neural Network for binary classification of 3D volumetric data.

    The architecture consists of a feature extraction backbone followed by a classifier head.
    The backbone uses a series of convolutional blocks (Conv3D -> BatchNorm3d -> ReLU)
    interspersed with MaxPool3d layers for spatial down-sampling. The classifier
    head is a multi-layer perceptron that maps the extracted features to a single
    logit for binary classification.

    The input size for the classifier's first linear layer is calculated dynamically
    based on the provided input shape, making the architecture flexible to changes
    in input dimensions or pooling layers.
    """
    def __init__(self, input_shape=(1, 32, 32, 16)):
        """
        Initializes the network architecture.

        Args:
            input_shape (tuple): The shape of the input tensor, expected as
                                 (channels, depth, height, width).
        """
        super(Conv3DNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Dynamically calculate the number of features after the convolutional layers
        with torch.no_grad():
            _dummy_input = torch.randn(1, *input_shape)
            _dummy_output = self.features(_dummy_input)
            in_features_for_fc = _dummy_output.view(-1).shape[0]

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(in_features_for_fc, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input batch of 3D tensors.

        Returns:
            torch.Tensor: The output logits from the classifier.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Custom3DDataset(Dataset):
    """
    A custom PyTorch Dataset for handling 3D image tensors and their corresponding labels.

    This class wraps tensor data and labels into a dataset compatible with a PyTorch
    DataLoader, which provides an iterator for easy batching, shuffling, and loading.
    """
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
    Stops training when a monitored metric has stopped improving.

    This callback monitors a given metric and, if no improvement is seen
    for a 'patience' number of epochs, training is stopped. It can also save
    the best model seen so far.
    """

    def __init__(self, patience=7, mode="min", verbose=False, delta=0, path='best_model.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation score improved. Default: 7
            mode (str): One of {"min", "max"}. In "min" mode, training will stop when the quantity
                        monitored has stopped decreasing; in "max" mode it will stop when the
                        quantity monitored has stopped increasing. Default: "min"
            verbose (bool): If True, prints a message for each validation score improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for saving the model checkpoint. Default: 'best_model.pt'
            trace_func (function): a function to use for printing messages. Default: print
        """
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
            self.val_score_min = -np.inf

    def __call__(self, score, model):
        """
        This method is called at the end of each validation epoch to check for early stopping criteria.
        """
        if self.mode == "min":
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
        """Saves model when monitored metric improves."""
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.val_score_min:.6f} --> {score:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
