import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        TODO: Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and biases with the correct shapes.
        self.params = {
            'weight': np.random.randn(in_features, out_features) / np.sqrt(2. / in_features),
            'bias': np.zeros((1, out_features))
        }
        self.grads = {'weight': None, 'bias': None}
        self.cache = {'input': None}
    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        TODO: Implement the forward pass.
        """
        self.cache['input'] = x
        return x @ self.params['weight'] + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        TODO: Implement the backward pass.
        """
        # dout.shape = (batch_size, out_features) or (batch_size, n_classes) 
        dw = self.cache['input'].T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.params['weight'].T
        self.grads['weight'] = dw
        self.grads['bias'] = db
        return dx

class ReLU(object):
    def __init__(self):
        self.cache = {'input': None}
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        TODO: Implement the forward pass.
        """
        self.cache['input'] = x
        return np.maximum(0, x)
    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        TODO: Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        # dout.shape = (batch_size, out_features)
        dx = dout * (self.cache['input'] > 0)
        return dx
    
class SoftMax(object):
    def __init__(self):
        self.output = None
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        TODO: Implement the forward pass using the Max Trick for numerical stability.
        """
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        TODO: Keep this in mind when implementing CrossEntropy's backward method.
        """
        return dout

class CrossEntropy(object):
    def __init__(self):
        self.cache = {'predictions': None, 'labels': None}
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        TODO: Implement the forward pass.
        """
        self.cache['predictions'] = x
        self.cache['labels'] = y
        m = x.shape[0]
        loss = -np.sum(y * np.log(x)) / m
        return loss
    def backward(self):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        TODO: Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        m = self.cache['labels'].shape[0]
        dx = (self.cache['predictions'] - self.cache['labels']) / m
        return dx # shape (batch_size, n_classes)
