"""Basic functions for neural networks."""

import function

import numpy as np

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
