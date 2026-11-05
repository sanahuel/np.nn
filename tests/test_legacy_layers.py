# Temporal: se elimina en la Fase 3.4, cuando test_layers.py y test_model.py lo sustituyan.
import numpy as np
import pytest

from nn.model import FCLayer
from gradcheck import numerical_grad, rel_error


@pytest.mark.parametrize("activation", ["relu", "sigmoid"])
def test_fclayer_gradients(activation):
    layer = FCLayer(5, 4, activation)
    x = np.random.randn(5, 1)
    R = np.random.randn(4, 1)
    if activation == "relu":  # alejar las preactivaciones del 0 (punto no derivable)
        layer.b[:] = np.where(np.abs(layer.W @ x) < 1e-2, 0.5, 0.0)
    loss = lambda: float(np.sum(layer.forward(x) * R))
    loss()
    dx, dW, db = layer.backward(R)
    assert rel_error(dx, numerical_grad(loss, x)) < 1e-5
    assert rel_error(dW, numerical_grad(loss, layer.W)) < 1e-5
    assert rel_error(db, numerical_grad(loss, layer.b)) < 1e-5


def test_softmax_layer_gradient_matches_cross_entropy():
    layer = FCLayer(5, 3, activation='softmax')
    x = np.random.randn(5, 1)
    y = np.array([[0.], [1.], [0.]])

    def loss():
        p = layer.forward(x)
        return float(-np.sum(y * np.log(p)))

    p = layer.forward(x)
    dx, dW, db = layer.backward(p - y)
    assert rel_error(dW, numerical_grad(loss, layer.W)) < 1e-5
    assert rel_error(db, numerical_grad(loss, layer.b)) < 1e-5
    assert rel_error(dx, numerical_grad(loss, x)) < 1e-5


def test_sigmoid_does_not_overflow():
    layer = FCLayer(1, 1, 'sigmoid')
    with np.errstate(over='raise'):
        out = layer.sigmoid(np.array([[-1e4], [1e4]]))
        layer.sigmoid(np.array([[-1e4], [1e4]]), derivative=True)
    assert np.allclose(out, [[0.0], [1.0]])
