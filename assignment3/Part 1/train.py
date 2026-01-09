from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path: Path):
    # Use a non-interactive backend to avoid GUI requirements.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val/Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs, label="Val/Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        # Transpose from (batch_size, seq_len, input_dim) to (seq_len, batch_size, input_dim)
        batch_inputs = batch_inputs.transpose(0, 1)
        optimizer.zero_grad(set_to_none=True)
        logit = model(batch_inputs)  # (batch_size, output_dim)
        loss = criterion(logit, batch_targets)
        loss.backward()
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        acc = accuracy(logit, batch_targets)
        batch_size = batch_targets.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        # Add more code here ...
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        # Transpose from (batch_size, seq_len, input_dim) to (seq_len, batch_size, input_dim)
        batch_inputs = batch_inputs.transpose(0, 1)
        logit = model(batch_inputs)  # (batch_size, output_dim)
        loss = criterion(logit, batch_targets)

        acc = accuracy(logit, batch_targets)
        batch_size = batch_targets.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(acc, batch_size)
        # Add more code here ...
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(config):
    if getattr(config, 'device', 'auto') != 'auto':
        device = config.device
    else:
        device = (
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Requested device=mps but MPS is not available in this PyTorch build.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested device=cuda but CUDA is not available.')

    print(f'Using device: {device}')
    # Initialize the model that we are going to use
    model = LSTM(input_dim=config.input_dim, hidden_dim=config.num_hidden, output_dim=config.num_classes)
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(input_length=config.input_length, total_len=config.data_size, one_hot=True)
    # Split dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(dataset, test_size=1-config.portion_train)
    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,)
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch, eta_min=1e-5)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(config.max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(model, train_dloader, optimizer, criterion, device, config)
        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        schedular.step()
        print(
            f'Epoch [{epoch+1}/{config.max_epoch}] '
            f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} | '
            f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}'
            )

    plot_path = Path(__file__).resolve().parent / "training_curves.png"
    plot_curves(train_losses, val_losses, train_accs, val_accs, plot_path)
    print(f"Saved curves to: {plot_path}")
    print('Done training.')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
                        help='Device to use: auto/cpu/mps/cuda')
    # Model params
    parser.add_argument('--input_length', type=int, default=19,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=5, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
