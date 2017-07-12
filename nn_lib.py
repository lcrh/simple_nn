"""Basic functions for neural networks."""

import function

import numpy as np

class Multiply(function.PointwiseFunction):
    """Pointwise multiply."""
    def execute(self, input_array):
        return np.prod(input_array, 0)

    def pointwise_gradient(self, index, input_array):
        return np.prod(np.delete(input_array, index, 0), 0)

class Divide(function.PointwiseFunction):
    """Pointwise divide."""
    def execute(self, input_array):
        assert len(input_array) == 2
        return input_array[0] / input_array[1]

    def pointwise_gradient(self, index, input_array):
        if index == 0:
            return 1 / input_array[1]
        elif index == 1:
            return -input_array[0] / (np.power(input_array[1], 2))
        else:
            assert False

class Add(function.PointwiseFunction):
    """Pointwise add."""
    def execute(self, input_array):
        return np.sum(input_array, 0)

    def pointwise_gradient(self, index, input_array):
        return np.ones(input_array[0].shape)

class Subtract(function.PointwiseFunction):
    """Pointwise subtract."""
    def execute(self, input_array):
        assert len(input_array) == 2
        return input_array[0] - input_array[1]

    def pointwise_gradient(self, index, input_array):
        if index == 0:
            return np.ones(input_array[0].shape)
        if index == 1:
            return -np.ones(input_array[1].shape)

class Power(function.PointwiseFunction):
    """Pointwise power function."""
    def execute(self, input_array):
        assert len(input_array) == 2
        return np.power(input_array[0], input_array[1])

    def pointwise_gradient(self, index, input_array):
        if index == 0:
            return input_array[1] * np.power(input_array[0], input_array[1] - 1)
        elif index == 1:
            return (np.power(input_array[0], input_array[1]) *
                    np.log(input_array[0]))
        else:
            assert False

class Exp(function.PointwiseFunction):
    """Pointwise exp function."""
    def execute(self, input_array):
        assert len(input_array) == 1
        return np.exp(input_array[0])

    def pointwise_gradient(self, index, input_array):
        assert index == 0
        return np.exp(input_array[0])

class Log(function.PointwiseFunction):
    """Pointwise natural logarithm."""
    def execute(self, input_array):
        assert len(input_array) == 1
        return np.log(input_array[0])

    def pointwise_gradient(self, index, input_array):
        assert index == 0
        return 1.0 / input_array[0]

class LogBase(function.FunctionBuilder):
    """Pointwise log function with arbitrary base. Second argument is base."""

    def __init__(self, graph, identifier):
        """Instantiates base change formula as function graph:
             log_x(y) = log(y) / log(x)
        """
        super(LogBase, self).__init__(graph, identifier)
        self._log_y = graph.add_function(identifier + "__log_y", Log)
        self._log_x = graph.add_function(identifier + "__log_x", Log)
        self._x_log_of_y = graph.add_function(identifier, Divide).set_inputs(
            self._log_y, self._log_x)
        self._input_map = [(self._log_x, 0), (self._log_y, 0)]

    @property
    def input_map(self):
        return self._input_map

    @property
    def output(self):
        return self._x_log_of_y

class MatMul(function.Function):
    """Function for matrix multiplication."""

    def get_shape(self, input_shapes):
        """(A, B) x (B, C) = (A, C)."""
        (rows_a, cols_a), (rows_b, cols_b) = input_shapes
        if cols_a != rows_b:
            raise ValueError(
                "Can't matrix multiply: (%d, ->%d) x (%d<-, %d)" % (
                    rows_a, cols_a, rows_b, cols_b))
        return (rows_a, cols_b)

    def execute(self, input_array):
        """Perform standard matrix multiplication."""
        return np.matmul(input_array[0], input_array[1])

    def gradient(self, index, input_array):
        """Compute gradient w.r.t. input_array[index] matrix."""
        (rows1, cols1), (rows2, cols2) = input_array[0].shape, input_array[1].shape
        assert cols1 == rows2

        grad = np.zeros(self.shape + input_array[index].shape)

        if index == 0:
            grad[np.arange(rows1), :, np.arange(rows1), :] = (
                input_array[1].transpose())
        elif index == 1:
            grad[:, np.arange(cols2), :, np.arange(cols2)] = (
                input_array[0])
        else:
            assert False
        return grad
