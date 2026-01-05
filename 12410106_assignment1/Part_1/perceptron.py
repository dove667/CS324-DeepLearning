import numpy as np
class Perceptron():
    def __init__(self, n_inputs, max_epochs=3, learning_rate=0.001, batch_size=32):
        self.n_inputs = n_inputs  # Fill in: Initialize number of inputs
        self.max_epochs = max_epochs  # Fill in: Initialize maximum number of epochs
        self.learning_rate = learning_rate # Fill in: Initialize learning rate
        self.batch_size = batch_size # Mini-batch size
        self.weights = np.zeros((self.n_inputs + 1, 1)) 

    def forward(self, input_vec): 
        return np.sign(input_vec @ self.weights).flatten()
    
    def train(self, training_inputs, labels):
        print('Start training...')
        # Add bias term to training inputs
        ones = np.ones((training_inputs.shape[0], 1))
        training_inputs_with_bias = np.concatenate((training_inputs, ones), axis=1)
        
        for epoch in range(self.max_epochs): 
            # Shuffle the data
            indices = np.arange(training_inputs_with_bias.shape[0])
            np.random.shuffle(indices)
            training_inputs_shuffled = training_inputs_with_bias[indices]
            labels_shuffled = labels[indices]
            
            misclassified_count = 0
            running_misclassified = 0

            loss = 0
            running_loss = 0

            # Process mini-batches
            for i in range(0, len(training_inputs_shuffled), self.batch_size):
                # Get mini-batch
                batch_inputs = training_inputs_shuffled[i:i+self.batch_size]
                batch_labels = labels_shuffled[i:i+self.batch_size]
                
                # Forward pass
                predictions = self.forward(batch_inputs)  
                
                # Calculate misclassified mask
                misclassified_mask = (predictions * batch_labels <= 0)

                running_misclassified += np.sum(misclassified_mask)
                misclassified_count += running_misclassified
                
                if np.any(misclassified_mask):
                    misclassified_inputs = batch_inputs[misclassified_mask]
                    misclassified_labels = batch_labels[misclassified_mask]

                    # calculate loss
                    running_loss = -np.sum(misclassified_labels * (misclassified_inputs @ self.weights).flatten())
                    loss += running_loss
                    # Compute gradient for misclassified samples
                    gradient = -np.sum(misclassified_labels[:, np.newaxis] * misclassified_inputs, axis=0)
                    
                    # Update weights using gradient descent
                    self.weights -= self.learning_rate * gradient[:, np.newaxis]
                print(f'Step{i+self.batch_size}/{training_inputs_with_bias.shape[0]}, loss:{running_loss:.5f}')

            print(f'Epoch {epoch + 1}/{self.max_epochs}, loss:{loss:.5f}, misclassify {misclassified_count} samples')
            
    def test(self, test_inputs, test_labels):
        ones = np.ones((test_inputs.shape[0], 1))
        test_inputs_with_bias = np.concatenate((test_inputs, ones), axis=1)
        test_predictions = self.forward(test_inputs_with_bias)
        misclassified_count = np.sum(test_predictions * test_labels <= 0)
        accuracy = 1 - misclassified_count / test_inputs.shape[0]
        print(f'Test accuracy: {accuracy:.2f}')