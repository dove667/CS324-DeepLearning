from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pytorch_mlp import MLP

def generate_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train)
    y_test_onehot = encoder.transform(y_test)
    return X_train, X_test, y_train_onehot, y_test_onehot

class MoonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size: int = 32, shuffle: bool = True, num_workers: int = 0,
                    pin_memory: bool = False):
    X_train, X_test, y_train, y_test = generate_data()
    train_ds = MoonDataset(X_train, y_train)
    test_ds = MoonDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 15
EVAL_FREQ_DEFAULT = 1

FLAGS = None

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    # targets: long class index (N,) or one-hot (N,C)
    if targets.dtype != torch.long or targets.ndim != 1:
        targets = targets.argmax(dim=1)
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train(dnn_hidden_units, learning_rate, max_epochs, eval_freq):
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    train_loader, test_loader = get_dataloaders(batch_size=64)
    model = MLP(2, dnn_hidden_units, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, running_acc, n_batches = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            y_idx = yb.argmax(dim=1) if (yb.dtype != torch.long or yb.ndim != 1) else yb

            logits = model(xb)
            loss = criterion(logits, y_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(logits, yb)
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)
        avg_train_acc = running_acc / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        if (epoch % eval_freq == 0) or (epoch == max_epochs):
            model.eval()
            test_loss_sum, test_acc_sum, t_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    y_idx = yb.argmax(dim=1) if (yb.dtype != torch.long or yb.ndim != 1) else yb
                    logits = model(xb)
                    loss = criterion(logits, y_idx)
                    test_loss_sum += loss.item()
                    test_acc_sum += accuracy(logits, yb)
                    t_batches += 1
            avg_test_loss = test_loss_sum / max(t_batches, 1)
            avg_test_acc = test_acc_sum / max(t_batches, 1)
            test_losses.append(avg_test_loss)

            print(f"Epoch {epoch:3d} | Train Loss {avg_train_loss:.4f} | "
                  f"Train Acc {avg_train_acc:.4f} | Test Loss {avg_test_loss:.4f} | "
                  f"Test Acc {avg_test_acc:.4f}")

    return train_losses, test_losses


def main(FLAGS):
    dnn_hidden_units = [int(x) for x in FLAGS.dnn_hidden_units.split(',')] if FLAGS.dnn_hidden_units else []
    train(dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)