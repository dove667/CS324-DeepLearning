from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cnn_model import CNN


LEARNING_RATE_DEFAULT = 1e-3
BATCH_SIZE_DEFAULT = 128
MAX_EPOCHS_DEFAULT = 100
WEIGHT_DECAY_DEFAULT = 1e-3
EVAL_FREQ_DEFAULT = 5
DATA_DIR_DEFAULT = './data/'

FLAGS = None

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.dtype != torch.long or targets.ndim != 1:
        targets = targets.argmax(dim=1)
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def get_dataloaders(batch_size, data_dir=DATA_DIR_DEFAULT):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    os.makedirs(data_dir, exist_ok=True)
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=test_transform
    )
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    return train_loader, test_loader

def train(learning_rate, max_epochs, batch_size, eval_freq, data_dir, weight_decay):
    print("Starting training...")
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    train_loader, test_loader = get_dataloaders(batch_size, data_dir)
    model = CNN(3,10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

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

        scheduler.step()

        avg_train_loss = running_loss / max(n_batches, 1)
        avg_train_acc = running_acc / max(n_batches, 1)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

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
            test_accuracies.append(avg_test_acc)

            print(f"Epoch {epoch:3d} | Train Loss {avg_train_loss:.4f} | "
                  f"Train Acc {avg_train_acc:.4f} | Test Loss {avg_test_loss:.4f} | "
                  f"Test Acc {avg_test_acc:.4f}")

    return train_losses, train_accuracies, test_losses, test_accuracies

def plot_curves(train_values, test_values, ylabel, eval_freq):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(train_values) + 1))
    eval_epochs = list(range(eval_freq, eval_freq * len(test_values) + 1, eval_freq))
    plt.figure()
    plt.plot(epochs, train_values, label='Train')
    plt.plot(eval_epochs, test_values, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f'{ylabel.lower()}_curve.png')
    plt.show()

def main(FLAGS):
    train_losses, train_accuracies, test_losses, test_accuracies = train(FLAGS.learning_rate, FLAGS.max_epochs, FLAGS.batch_size, 
                                                                         FLAGS.eval_freq, FLAGS.data_dir, FLAGS.weight_decay)
    plot_curves(train_losses, test_losses, 'Loss', FLAGS.eval_freq)
    plot_curves(train_accuracies, test_accuracies, 'Accuracy', FLAGS.eval_freq)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_epochs', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default= EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--weight_decay', type = float, default = WEIGHT_DECAY_DEFAULT,
                      help='Weight decay (L2 penalty)')
  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS)