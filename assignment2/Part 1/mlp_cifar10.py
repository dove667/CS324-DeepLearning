import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE_DEFAULT = 128
LEARNING_RATE_DEFAULT = 0.1
MAX_EPOCHS_DEFAULT = 100
WEIGHT_DECAY_DEFAULT = 5e-4
EVAL_FREQ_DEFAULT = 10
DATA_DIR_DEFAULT = '../data/'

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
        transforms.ToTensor(),
        transforms.Normalize(*stats)
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

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 10)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.classifier(x)
        return x


def train(learning_rate, max_epochs, batch_size, eval_freq, data_dir):

    train_loader, test_loader = get_dataloaders(batch_size, data_dir)
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY_DEFAULT)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.001)
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

        scheduler.step()

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
    train(FLAGS.learning_rate, FLAGS.max_steps, FLAGS.batch_size, FLAGS.eval_freq, FLAGS.data_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main(FLAGS)

  