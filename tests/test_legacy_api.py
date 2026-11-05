import numpy as np
from sklearn.datasets import make_moons, load_digits

from nn.model import NN


def _legacy_onehot(y, n):
    return np.eye(n)[y].reshape(len(y), n, 1)          # (N, n, 1)


def test_moons_legacy_shapes(capsys):
    X, y = make_moons(n_samples=400, noise=0.1, random_state=0)
    X = X.reshape(len(X), 2, 1)                          # (N, 2, 1)
    Y = _legacy_onehot(y, 2)
    nn = NN(layers=[2, 16, 2], function='relu')
    nn.train(X, Y, epochs=30, lr=0.01)
    nn.test(X, Y)
    out = capsys.readouterr().out
    assert "accuracy" in out.lower()


def test_digits_legacy_shapes():
    X, y = load_digits(return_X_y=True)
    X = X.reshape(len(X), 64, 1)
    Y = _legacy_onehot(y, 10)
    nn = NN(layers=[64, 128, 64, 32, 10], function='relu')
    nn.train(X[:500], Y[:500], epochs=5, lr=0.0001)
    nn.test(X[500:700], Y[500:700])


def test_moons_legacy_min_accuracy():
    # umbral = 0.9 x la peor accuracy observada en 3 semillas (0.97)
    X, y = make_moons(n_samples=400, noise=0.1, random_state=0)
    X = X.reshape(len(X), 2, 1)
    Y = _legacy_onehot(y, 2)
    nn = NN(layers=[2, 16, 2], function='relu')
    nn.train(X, Y, epochs=30, lr=0.01)
    correct = 0
    for x, t in zip(X, Y):
        for layer in nn.layers:
            x = layer.forward(x)
        correct += int(np.argmax(x) == np.argmax(t))
    assert correct / len(X) >= 0.87
