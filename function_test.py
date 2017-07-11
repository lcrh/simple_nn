"""Tests for function.py"""

import function
import numpy as np

class Square(function.PointwiseFunction):
    """Square input tensor."""

    def execute(self, input_array):
        return np.power(input_array[0], 2)

    def pointwise_gradient(self, index, input_array):
        return 2 * input_array[0]

class Multiplier(function.PointwiseFunction):
    """Multiply input tensors."""

    def execute(self, input_array):
        return np.product(input_array, 0)

    def pointwise_gradient(self, index, input_array):
        return np.product(np.delete(input_array, index, 0), 0)

class SquareMultiplier(function.FunctionBuilder):
    """FunctionBuilder test. Computes x^2 * y."""
    def __init__(self, graph, identifier):
        self._square = graph.add_function(identifier + "_sq", Square)
        self._mult = graph.add_function(identifier, Multiplier)
        self._mult.set_input(0, self._square)
        self._input_map = [(self._square, 0), (self._mult, 1)]

    @property
    def input_map(self):
        """Returns mapping from input indices to (fun, fun_index)."""
        return self._input_map

    @property
    def output(self):
        """Returns the top-level output function of this builder."""
        return self._mult

class Sum(function.Function):
    """Performs sum over input elements."""

    def execute(self, input_array):
        """Compute sum."""
        return np.sum(
            np.sum(inp for inp in input_array))

    def gradient(self, index, input_array):
        """Compute gradient w.r.t. input_array[index]."""
        return np.array([np.ones(input_array[index].shape)])

    @property
    def shape(self):
        """Return shape."""
        return (1,)

def get_test_graph():
    """Create example graph."""
    graph = function.FunctionGraph()
    graph.add_function("sqmult1", SquareMultiplier).set_inputs(
        graph.add_function("const1", function.Constant, (3, 2)),
        graph.add_function("sqmult2", SquareMultiplier).set_inputs(
            graph.add_function("var1", function.Variable, (3, 2)),
            graph.add_function("var2", function.Variable, (3, 2))))
    return graph

def test_build_graph():
    """Test basic graph construction."""
    graph = get_test_graph()

    # Try looking up functions by identifier.
    assert isinstance(graph.lookup("sqmult1"), Multiplier)
    assert isinstance(graph.lookup("const1"), function.Constant)
    assert isinstance(graph.lookup(["const1", "var2"])[1], function.Variable)
    assert graph.lookup("const1").identifier == "const1"

    # Try graph navigation functions.
    assert list(graph.lookup("var2").outputs())[0] == graph.lookup("sqmult2")

    # Test no error w/o cycle
    assert len(list(graph.traverse_topological())) == 7

    # Test error on cycle.
    graph.add_function("cyclic", SquareMultiplier).set_inputs(
        graph.lookup("var1"),
        graph.lookup("cyclic"))
    try:
        assert not list(graph.traverse_topological())
        assert False  # This should abort with an exception
    except function.Error:
        pass

def test_compute_result():
    """Test computing results."""
    graph = get_test_graph()
    context = function.ExecutionContext(
        graph,
        {"var1": [0, 1, 2, 3, 4, 5],
         "var2": [1, 2, 3, 4, 5, 6]},
        np.ones)
    val = context.compute("sqmult1")
    # Function is const1^2 * (var1^2 * var2)
    # (1, 1, 1, 1, 1, 1)^2 * (0, 1, 2, 3, 4, 5)^2 * (1, 2, 3, 4, 5, 6)
    expected = np.array([0.0, 2.0, 12.0, 36.0, 80.0, 150.0]).reshape(3, 2)
    assert (expected == val).all()

def test_pointwise_gradient():
    """Test PointwiseFunction gradient computation."""
    graph = function.FunctionGraph()
    graph.add_function("sq", Square).set_inputs(
        graph.add_function("var1", function.Variable, (3,)))
    context = function.ExecutionContext(
        graph,
        {"var1": [1, 2, 3]})
    assert (context.compute("sq") == [1, 4, 9]).all()
    grad = context.gradient("sq", "var1")
    expected = [
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
    ]
    assert (grad == expected).all()

def test_complex_gradient():
    """Test computation of complex gradients."""
    graph = function.FunctionGraph()
    graph.add_function("sq", Square).set_inputs(
        graph.add_function("sum", Sum).set_inputs(
            graph.add_function("sqmult", SquareMultiplier).set_inputs(
                graph.add_function("var1", function.Variable, (2,)),
                graph.add_function("var2", function.Variable, (2,)))))

    context = function.ExecutionContext(
        graph,
        {"var1": [1, 2],
         "var2": [3, 4]})

    dsqmultdvar1 = context.gradient("sqmult", "var1")
    expected_dsqmultdvar1 = [
        [6, 0],
        [0, 16]
    ]
    assert (dsqmultdvar1 == expected_dsqmultdvar1).all()

    dsqmultdvar2 = context.gradient("sqmult", "var2")
    expected_dsqmultdvar2 = [
        [1, 0],
        [0, 4]
    ]
    assert (dsqmultdvar2 == expected_dsqmultdvar2).all()

    dsumdvar1 = context.gradient("sum", "var1")
    expected_dsumdvar1 = [
        [6, 16]
    ]
    assert (dsumdvar1 == expected_dsumdvar1).all()

    dsumdvar2 = context.gradient("sum", "var2")
    expected_dsumdvar2 = [
        [1, 4]
    ]
    assert (dsumdvar2 == expected_dsumdvar2).all()

    dsqdvar1 = context.gradient("sq", "var1")
    expected_dsqdvar1 = [
        [228, 608]
    ]
    assert (dsqdvar1 == expected_dsqdvar1).all()

    dsqdvar2 = context.gradient("sq", "var2")
    expected_dsqdvar2 = [
        [38, 152]
    ]
    assert (dsqdvar2 == expected_dsqdvar2).all()
