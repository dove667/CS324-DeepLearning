from modules import * 

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        self.layers = []
        for hidden in n_hidden:
            self.layers.append(Linear(n_inputs, hidden))
            self.layers.append(ReLU())
            n_inputs = hidden
        self.layers.append(Linear(n_inputs, n_classes))
        self.layers.append(SoftMax())

    def forward(self, x):
        out = x  
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update_params(self, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.params['weight'] -= learning_rate * layer.grads['weight']
                layer.params['bias'] -= learning_rate * layer.grads['bias']


