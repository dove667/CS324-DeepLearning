import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score 

from numpy_mlp import MLP  
from modules import CrossEntropy

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 15
EVAL_FREQ_DEFAULT = 1
GD_MODE = 0
BATCH_SIZE = 32

def accuracy(predictions, targets):
    return accuracy_score(np.argmax(targets, axis=1), np.argmax(predictions, axis=1)) * 100.0

def generate_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train)
    y_test_onehot = encoder.transform(y_test)
    return X_train, X_test, y_train_onehot, y_test_onehot

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current = 0
        return self
        
    def __next__(self):
        if self.current >= self.num_samples:
            raise StopIteration
        start = self.current
        end = min(self.current + self.batch_size, self.num_samples)
        batch_idx = self.indices[start:end]
        self.current = end
        return self.X[batch_idx], self.y[batch_idx]
        
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
def train(dnn_hidden_units, learning_rate, max_epochs, eval_freq, gd_mode, batch_size):
    # Set seed only for data generation to ensure reproducibility
    np.random.seed(42)
    X_train, X_test, y_train, y_test = generate_data()
    
    # Reset seed for training to allow different shuffles per epoch
    np.random.seed(None)
    
    mlp = MLP(n_inputs=2, n_hidden=dnn_hidden_units, n_classes=2)
    criteria = CrossEntropy()
    
    # Create test dataloader with shuffle=False for evaluation
    testloader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    if gd_mode == 0:  # batch mode
        for epoch in range(max_epochs):
            output = mlp.forward(X_train)
            loss = criteria.forward(output, y_train)
            train_accuracy = accuracy(output, y_train)
            mlp.backward(criteria.backward())
            mlp.update_params(learning_rate)
            
            if epoch % eval_freq == 0 or epoch == max_epochs - 1:
                test_loss_sum = 0.0
                test_acc_sum = 0.0
                test_batch_count = 0
                
                for X_batch, y_batch in testloader:
                    test_output = mlp.forward(X_batch)
                    test_loss_sum += criteria.forward(test_output, y_batch)
                    test_acc_sum += accuracy(test_output, y_batch)
                    test_batch_count += 1
                
                test_loss = test_loss_sum / test_batch_count
                test_accuracy = test_acc_sum / test_batch_count
                
                print(f"Epoch: {epoch+1:3d} | Train Loss: {loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
        print("Training complete!")

    elif gd_mode == 1:  # mini-batch mode
        trainloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
        for epoch in range(max_epochs):
            epoch_train_loss = 0.0
            epoch_train_accuracy = 0.0
            batch_count = 0
            
            for X_batch, y_batch in trainloader:
                output = mlp.forward(X_batch)
                loss = criteria.forward(output, y_batch)
                mlp.backward(criteria.backward())
                mlp.update_params(learning_rate)
                
                epoch_train_loss += loss
                epoch_train_accuracy += accuracy(output, y_batch)
                batch_count += 1
                
            avg_train_loss = epoch_train_loss / batch_count
            avg_train_accuracy = epoch_train_accuracy / batch_count
            
            if epoch % eval_freq == 0 or epoch == max_epochs - 1:
                test_loss_sum = 0.0
                test_acc_sum = 0.0
                test_batch_count = 0
                
                for X_batch, y_batch in testloader:
                    test_output = mlp.forward(X_batch)
                    test_loss_sum += criteria.forward(test_output, y_batch)
                    test_acc_sum += accuracy(test_output, y_batch)
                    test_batch_count += 1

                test_loss = test_loss_sum / test_batch_count
                test_accuracy = test_acc_sum / test_batch_count

                print(f"Epoch: {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
        print("Training complete!")

    elif gd_mode == 2:  # SGD
        trainloader = DataLoader(X_train, y_train, batch_size=1, shuffle=True)
        for epoch in range(max_epochs):
            epoch_train_loss = 0.0
            epoch_train_accuracy = 0.0
            batch_count = 0
            
            for X_batch, y_batch in trainloader:
                output = mlp.forward(X_batch)
                loss = criteria.forward(output, y_batch)
                mlp.backward(criteria.backward())
                mlp.update_params(learning_rate)
                
                epoch_train_loss += loss
                epoch_train_accuracy += accuracy(output, y_batch)
                batch_count += 1
                
            avg_train_loss = epoch_train_loss / batch_count
            avg_train_accuracy = epoch_train_accuracy / batch_count
            
            if epoch % eval_freq == 0 or epoch == max_epochs - 1:
                test_loss_sum = 0.0
                test_acc_sum = 0.0
                test_batch_count = 0
                
                for X_batch, y_batch in testloader:
                    test_output = mlp.forward(X_batch)
                    test_loss_sum += criteria.forward(test_output, y_batch)
                    test_acc_sum += accuracy(test_output, y_batch)
                    test_batch_count += 1

                test_loss = test_loss_sum / test_batch_count
                test_accuracy = test_acc_sum / test_batch_count

                print(f"Epoch: {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_accuracy:.2f}% | "
                      f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}%")
        print("Training complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--gd_mode', type=int, default=GD_MODE,
                        help='0 for batch mode, 1 for mini-batch mode, 2 for SGD')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for mini-batch mode')
    FLAGS = parser.parse_known_args()[0]

    dnn_hidden_units = [int(x) for x in FLAGS.dnn_hidden_units.split(',')] if FLAGS.dnn_hidden_units else []

    train(dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq, FLAGS.gd_mode, FLAGS.batch_size)

if __name__ == '__main__':
    main()
