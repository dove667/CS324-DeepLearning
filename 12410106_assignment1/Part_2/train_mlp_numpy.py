import argparse
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score 

from mlp_numpy import MLP  
from modules import CrossEntropy
np.random.seed(42)
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 15 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 1

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    return accuracy_score(np.argmax(targets, axis=1), np.argmax(predictions, axis=1)) * 100.0

def generate_data():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    
def train(dnn_hidden_units, learning_rate, max_epochs, eval_freq):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    np.random.seed(42)
    X_train, X_test, y_train, y_test = generate_data()
    trainloader = DataLoader(X_train, y_train)
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here
    mlp = MLP(n_inputs=2, n_hidden=dnn_hidden_units, n_classes=2)
    criteria = CrossEntropy()
    
    for epoch in range(max_epochs):
        epoch_train_loss = 0
        batch_count = 0
        for step, (X_batch, y_batch) in enumerate(trainloader):
            output = mlp.forward(X_batch)
            loss = criteria.forward(output, y_batch)
            mlp.backward(criteria.backward())
            mlp.update_params(learning_rate)
            epoch_train_loss += loss
            batch_count += 1
        avg_train_loss = epoch_train_loss / batch_count
        
        print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}")
        if epoch % eval_freq == 0 or epoch == max_epochs - 1:
            test_output = mlp.forward(X_test)
            test_loss = criteria.forward(test_output, y_test)
            test_accuracy = accuracy(test_output, y_test)
            print(f"Epoch: {epoch+1}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}")
    print("Training complete!")

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]

    dnn_hidden_units = [int(x) for x in FLAGS.dnn_hidden_units.split(',')] if FLAGS.dnn_hidden_units else []

    train(dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
