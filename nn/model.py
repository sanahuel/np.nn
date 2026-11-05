import numpy as np

class FCLayer:
    def __init__(self, n_input, n_output, activation='relu'):
        if activation == 'relu':
            std = np.sqrt(2.0 / n_input)                # He (Kaiming) initialization
        else:
            std = np.sqrt(2.0 / (n_input + n_output))   # Xavier/Glorot (sigmoid, tanh, softmax)
        self.W = np.random.randn(n_output, n_input) * std
        self.b = np.zeros((n_output, 1))
        self.activation = activation
        self.cache = None
        self.cache_act = None

    def __repr__(self):
        return f'W: {self.W.shape}, b: {self.b.shape} with activation: {self.activation}'

    def relu(self, x, derivative = False):
        if derivative: return np.where(x>0, 1, 0)
        return np.maximum(0, x)

    def sigmoid(self, x, derivative = False):
        x = np.clip(x, -500, 500)
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s) if derivative else s

    def softmax(self, x):
        exps = np.exp(x - x.max(axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    def forward(self, x):
        self.cache = x
        self.cache_act = self.W @ x + self.b
        if self.activation == 'relu':
            x = self.relu(self.W @ x + self.b)
        elif self.activation == 'sigmoid':
            x = self.sigmoid(self.W @ x + self.b)
        elif self.activation == 'softmax':
            x = self.softmax(self.W @ x + self.b)

        return x

    def backward(self, delta):
        if self.activation == 'relu':
            delta = delta * self.relu(self.cache_act, derivative=True)
        elif self.activation == 'sigmoid':
            delta = delta * self.sigmoid(self.cache_act, derivative=True)
        elif self.activation == 'softmax':
            # NN.train passes (p - y), which is already dL/dz for softmax + cross-entropy.
            # It must not be multiplied by the softmax derivative again.
            pass

        dW = delta @ self.cache.T
        db = np.sum(delta, axis=1, keepdims=True)
        dx = self.W.T @ delta

        return dx, dW, db

class NN:
    def __init__(self, layers, function='relu'):
        self.structure = layers
        self.input_size = layers[0]
        self.function = function.lower()
        if self.function not in ('relu', 'sigmoid'):
            raise ValueError(f"function must be 'relu' or 'sigmoid', got {function!r}")
        self.layers = [FCLayer(n_input=layers[i], n_output=layers[i + 1], activation=self.function)
                       for i in range(len(layers)-2)]
        self.layers.append(FCLayer(n_input=layers[-2], n_output=layers[-1], activation='softmax'))

    def __repr__(self):
        lines = [f'NN structure: {self.structure}'] + [f'  {layer!r}' for layer in self.layers]
        return '\n'.join(lines)

    def train(self, X, Y, epochs, lr=0.01):
        print(f'Training for {epochs} epochs...')
        for epoch in range(epochs):
            n_correct = 0
            for x_input, y in zip(X, Y):
                # forward
                x = x_input
                for layer in self.layers:
                    x = layer.forward(x)

                # count for accuracy
                n_correct += int(np.argmax(x) == np.argmax(y))

                # backpropagation
                loss_gradient = x - y
                for layer in reversed(self.layers):
                    loss_gradient, dW, db = layer.backward(loss_gradient)
                    # update the parameters
                    layer.W -= lr * dW
                    layer.b -= lr * db

            # print accuracy for epoch
            print(f'Epoch {epoch} accuracy: {n_correct / len(Y) * 100: .3f}%')

    def test(self, X, Y):
        n_correct = 0
        for x, y in zip(X, Y):
            for layer in self.layers:
                x = layer.forward(x)
            
            # count for accuracy
            n_correct += int(np.argmax(x) == np.argmax(y))

        print(f'Test accuracy: {n_correct / len(Y) * 100: .3f}%')
