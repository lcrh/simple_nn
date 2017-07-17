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

    def __init__(self, min_input=1e-15):
        """Min_input replaces zeros in input to avoid log(0)."""
        super(Log, self).__init__()
        self._min_input = min_input

    def execute(self, input_array):
        assert len(input_array) == 1
        return np.log(np.maximum(input_array[0], self._min_input))

    def pointwise_gradient(self, index, input_array):
        assert index == 0
        arr = np.maximum(input_array[0], self._min_input)
        return np.where(input_array[0] > self._min_input,
                        1 / arr, 0)

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

class Negate(function.PointwiseFunction):
    """Pointwise negation."""
    def execute(self, input_array):
        assert len(input_array) == 1
        return -input_array[0]

    def pointwise_gradient(self, index, input_array):
        assert index == 0
        return -np.ones(input_array[0].shape)

class Identity(function.PointwiseFunction):
    """Identity function."""
    def execute(self, input_array):
        assert len(input_array) == 1
        return np.array(input_array[0])

    def pointwise_gradient(self, index, input_array):
        return np.ones(input_array[0].shape)

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

class ReLU(function.PointwiseFunction):
    """Pointwise Rectangular linear activation unit."""

    def execute(self, input_array):
        assert len(input_array) == 1
        return np.maximum(input_array[0], 0)

    def pointwise_gradient(self, index, input_array):
        assert len(input_array) == 1
        return np.where(input_array[0] >= 0, 0.0, 1.0)

class Sigmoid(function.PointwiseFunction):
    """Pointwise sigmoid activation function."""

    def execute(self, input_array):
        assert len(input_array) == 1
        return 1/(1 + np.exp(-input_array[0]))

    def pointwise_gradient(self, index, input_array):
        exp_term = np.exp(input_array[0])
        return exp_term / np.power(exp_term + 1, 2)

class FullyConnected(function.FunctionBuilder):
    """Fully connected layer with bias."""

    def __init__(self, graph, identifier,
                 rows, input_cols, output_cols,
                 activation_function, bias=True):
        self._weights = graph.add_parameter(identifier + "_weights",
                                            (input_cols, output_cols))
        self._bias = graph.add_parameter(identifier + "_bias",
                                         (rows, output_cols))

        self._matmul = graph.add_function(
            identifier + "_matmul", MatMul).set_input(1, self._weights)

        self._add = graph.add_function(
            identifier + "_out", Add).set_inputs(self._bias, self._matmul)

        self._input_map = [(self._matmul, 0)]

        self._act = graph.add_function(
            identifier, activation_function).set_inputs(self._add)

    @property
    def input_map(self):
        return self._input_map

    @property
    def output(self):
        return self._act

class AddConstant(function.PointwiseFunction):
    """Add a constant value."""

    def __init__(self, value):
        super(AddConstant, self).__init__()
        self._value = np.array(value)

    def execute(self, input_array):
        assert len(input_array) == 1
        return input_array[0] + self._value

    def pointwise_gradient(self, index, input_array):
        assert index == 0
        return np.ones(input_array[0].shape)

class CrossEntropy(function.FunctionBuilder):
    """Cross entropy loss between predicted binary class probability input[0] 
    and actual binary class probabiltiy input[1]."""

    def __init__(self, graph, identifier):
        """Create cross entropy loss function graph:
            -input[1] * log(input[0]) - (1 - input[1]) * log(1 - input[0])
        """
        input0 = graph.add_function(identifier + "_inp0", Identity)
        input1 = graph.add_function(identifier + "_inp1", Identity)
        minus_input0 = graph.add_function(
            identifier + "_minus_inp0", Negate).set_inputs(input0)
        minus_input1 = graph.add_function(
            identifier + "_minus_inp1", Negate).set_inputs(input1)

        one_minus_input0 = graph.add_function(
            identifier + "_one_minus_0", AddConstant, 1.0).set_inputs(
                minus_input0)
        one_minus_input1 = graph.add_function(
            identifier + "_one_minus_1", AddConstant, 1.0).set_inputs(
                minus_input1)

        self._output = graph.add_function(identifier, Subtract).set_inputs(
            graph.add_function(identifier + "_mul1", Multiply).set_inputs(
                minus_input1,
                graph.add_function(identifier + "_log1", Log).set_inputs(
                    input0)),
            graph.add_function(identifier + "_mul2", Multiply).set_inputs(
                one_minus_input1,
                graph.add_function(identifier + "_log2", Log).set_inputs(
                    one_minus_input0)))
        self._input_map = [(input0, 0), (input1, 0)]

    @property
    def input_map(self):
        return self._input_map

    @property
    def output(self):
        return self._output
