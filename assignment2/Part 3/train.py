from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

def train(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes).to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # batch_inputs: (batch_size, seq_length,)
        # batch_targets: (batch_size,)
        model.train()
        # Convert inputs to one-hot encoding
        batch_inputs = batch_inputs.to(device).long()
        batch_inputs = torch.nn.functional.one_hot(batch_inputs, config.input_dim).float()
        batch_targets = batch_targets.to(device).long()
        logit = model(batch_inputs, batch_first=True)  
        # logit (batch_size, num_classes)
        loss = criterion(logit, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        loss = loss.item()
        accuracy = (logit.argmax(dim=1) == batch_targets).float().mean().item()
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

        if step >= config.train_steps:
            break

    print('Done training.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=200, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()

    train(config)