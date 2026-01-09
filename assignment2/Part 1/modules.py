import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.params = {
            'weight': np.random.randn(in_features, out_features) * np.sqrt(2. / in_features),
            'bias': np.zeros((1, out_features))
        }
        self.grads = {'weight': None, 'bias': None}
        self.cache = {'input': None}

    def forward(self, x):
        self.cache['input'] = x
        return x @ self.params['weight'] + self.params['bias']

    def backward(self, dout):
        input = self.cache.get('input')
        if input is None:
            raise ValueError("Linear.backward called before forward pass; no input cached.")
        dw = input.T @ dout
        db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.params['weight'].T
        self.grads['weight'] = dw
        self.grads['bias'] = db
        return dx

class ReLU(object):

    def __init__(self):
        self.cache = {'input': None}

    def forward(self, x):
        self.cache['input'] = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        inp = self.cache.get('input')
        if inp is None:
            raise ValueError("ReLU.backward called before forward pass; no input cached.")
        dx = dout * (inp > 0)
        return dx
    
class SoftMax(object):

    def __init__(self):
        self.output = None

    def forward(self, x):
        shift_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shift_x)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dout):
        return dout

class CrossEntropy(object):

    def __init__(self):
        self.cache = {'predictions': None, 'labels': None}

    def forward(self, x, y):
        self.cache['predictions'] = x
        self.cache['labels'] = y
        m = x.shape[0]
        loss = -np.sum(y * np.log(x)) / m
        return loss
    
    def backward(self):
        labels = self.cache.get('labels')
        predictions = self.cache.get('predictions')
        if labels is None or predictions is None:
            raise ValueError("CrossEntropy.backward called before forward pass; no data cached.")
        m = labels.shape[0]
        dx = (predictions - labels) / m
        return dx 
