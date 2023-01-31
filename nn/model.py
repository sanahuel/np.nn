import numpy as np


class Layer:
    def __init__(self, n_input, n_output):
        self.W = np.random.uniform(-0.5, 0.5, (n_output, n_input))
        self.b = np.zeros((n_output, 1))


class NN:
    def __init__(self, layers, function='relu'):
        self.structure = layers
        self.input_size = layers[0]
        self.layers = [Layer(n_input=layers[i], n_output=layers[i + 1]) for i in range(len(layers)-1)]
        self.function = function.lower()

    def __repr__(self):
        return f'NN structure: {self.structure}'

    def sigmoid(self, x, derivative = False):
        np.clip(x, -500, 500)
        if derivative: return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1 / (1 + np.exp(-x))

    def ReLU(self, x, derivative = False):
        if derivative: return np.where(x>0, 1, 0)
        return np.maximum(0, x)

    def softmax(self,x, derivative = False):
        exps = np.exp(x - x.max())
        if derivative: return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def train(self, X, Y, epochs, lr=0.01):
        print(f'Training for {epochs} epochs...')
        for epoch in range(epochs):
            n_correct = 0
            for x_input, y in zip(X, Y):
                # forward
                x = x_input
                x_inter = [0] * len(self.layers)
                for i, layer in enumerate(self.layers):
                    if i < (len(self.layers)-1):
                        if self.function == 'relu':
                            x = self.ReLU(layer.W @ x + layer.b)
                        else:
                            x = self.sigmoid(layer.W @ x + layer.b)
                    else:
                        x = self.softmax(layer.W @ x + layer.b)
                    x_inter[i] = x

                # count for accuracy
                n_correct += int(np.argmax(x) == np.argmax(y))

                # backpropagation
                delta = x - y
                for i, layer in reversed(list(enumerate(self.layers))):
                    layer.W += -lr * delta @ np.transpose(x_input if i==0 else x_inter[i-1])
                    layer.b += -lr * delta
                    if i==0: continue
                    if i == len(self.layers)-1:
                        delta = np.transpose(layer.W) @ delta * self.softmax(x_inter[i - 1], derivative=True)
                    else:
                        if self.function == 'relu':
                            delta = np.transpose(layer.W) @ delta * self.ReLU(x_inter[i-1], derivative=True)
                        else:
                            delta = np.transpose(layer.W) @ delta * self.sigmoid(x_inter[i-1], derivative=True)

            # print accuracy for epoch
            print(f'Epoch {epoch} accuracy: {n_correct / len(Y) * 100: .3f}%')

    def test(self, X, Y):
        n_correct = 0
        for x, y in zip(X, Y):
            for idx, layer in enumerate(self.layers):
                if self.function == 'relu':
                    x = self.ReLU(np.dot(layer.W, x) + layer.b)
                else:
                    x = self.sigmoid(np.dot(layer.W, x) + layer.b)

            # count for accuracy
            n_correct += int(np.argmax(x) == np.argmax(y))

        print(f'Test accuracy: {n_correct / len(Y) * 100: .3f}%')
