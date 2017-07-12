"""Tests for nn_lib.py."""

import math

import numpy as np

import function
import nn_lib

def test_basic_funs_simple():
    """Build (x+y)^y / (y * x)."""
    graph = function.FunctionGraph()
    graph.add_variable("x", (1, 1))
    graph.add_variable("y", (1, 1))
    graph.add_function("div", nn_lib.Divide).set_inputs(
        graph.add_function("pow", nn_lib.Power).set_inputs(
            graph.add_function("add", nn_lib.Add).set_inputs("x", "y"), "y"),
        graph.add_function("mul", nn_lib.Multiply).set_inputs("x", "y"))

    x, y = 3.0, 4.0
    context = function.ExecutionContext(
        graph, {"x": [[x]], "y": [[y]]})
    comp_fun = lambda x, y: pow(x + y, y) / (x * y)
    comp_got = context.compute("div")
    assert np.allclose(comp_got, [[comp_fun(x, y)]])

    # Gradient functions courtesy of wolfram alpha:
    pow_grad_x = lambda x, y: y * pow(x + y, y - 1)
    pow_grad_y = lambda x, y: pow(x + y, y) * (
        y / (x + y) + math.log(x + y))

    pow_grad_x_got = context.gradient("pow", "x")
    pow_grad_x_want = [[[[pow_grad_x(x, y)]]]]
    pow_grad_y_got = context.gradient("pow", "y")
    pow_grad_y_want = [[[[pow_grad_y(x, y)]]]]

    print(pow_grad_x_got)
    print("vs")
    print(pow_grad_x_want)
    assert np.allclose(pow_grad_x_got, pow_grad_x_want)
    print(pow_grad_y_got)
    print("vs")
    print(pow_grad_y_want)
    assert np.allclose(pow_grad_y_got, pow_grad_y_want)

    x_grad = lambda x, y: (
        (x * (y - 1) - y) * pow(x + y, y - 1) / (x * x * y))
    grad_x_got = context.gradient("div", "x")
    grad_x_want = [[[[x_grad(x, y)]]]]
    print(grad_x_got)
    print("vs:")
    print(grad_x_want)
    assert np.allclose(grad_x_got, grad_x_want)

    y_grad = lambda x, y: (
        (pow(x + y, y - 1) *
         (y * (x + y) * math.log(x + y) - x + (y - 1) * y)) /
        (x * y * y))

    grad_y_got = context.gradient("div", "y")
    grad_y_want = [[[[y_grad(x, y)]]]]
    print(grad_y_got)
    print("vs:")
    print(grad_y_want)
    assert np.allclose(grad_y_got, grad_y_want)

def test_basic_funs_log():
    """Build exp(log(x + y) - log_x(y))."""
    graph = function.FunctionGraph()
    graph.add_variable("x", (1, 1))
    graph.add_variable("y", (1, 1))

    graph.add_function("exp", nn_lib.Exp).set_inputs(
        graph.add_function("sub", nn_lib.Subtract).set_inputs(
            graph.add_function("log", nn_lib.Log).set_inputs(
                graph.add_function("add", nn_lib.Add).set_inputs("x", "y")),
            graph.add_function("log_x", nn_lib.LogBase).set_inputs("x", "y")))
    x, y = 3.0, 4.0
    context = function.ExecutionContext(
        graph, {"x": [[x]], "y": [[y]]})

    comp_got = context.compute("exp")
    comp_fun = lambda x, y: math.exp(math.log(x + y) - math.log(y, x))
    assert np.allclose(comp_got, [[[[comp_fun(x, y)]]]])

    def x_grad(x, y):
        """Gradient of exp w.r.t. x."""
        term1 = pow(y, -1 / math.log(x))
        term2 = (x + y) * math.log(y) + x * pow(math.log(x), 2)
        term3 = x * pow(math.log(x), 2)
        return (term1 * term2) / term3

    def y_grad(x, y):
        """Gradient of exp w.r.t. y."""
        term1 = pow(y, (-1 / math.log(x)) - 1)
        term2 = y * math.log(x) - x - y
        term3 = math.log(x)
        return (term1 * term2) / term3

    x_grad_got = context.gradient("exp", "x")
    x_grad_want = [[[[x_grad(x, y)]]]]
    assert np.allclose(x_grad_got, x_grad_want)

    y_grad_got = context.gradient("exp", "y")
    y_grad_want = [[[[y_grad(x, y)]]]]
    assert np.allclose(y_grad_got, y_grad_want)

def test_matmul_simple():
    """Simple matrix multiplication 1x2 x 2x1 = 1x1."""
    graph = function.FunctionGraph()
    graph.add_function("mul", nn_lib.MatMul).set_inputs(
        graph.add_variable("var1", (1, 2)),
        graph.add_variable("var2", (2, 1)))

    context = function.ExecutionContext(
        graph, {"var1": [[1, 2]], "var2": [[3], [4]]})

    comp_got = context.compute("mul")
    print(comp_got)
    comp_want = [[11]]
    assert (comp_got == comp_want).all()
    v1_grad = context.gradient("mul", "var1")
    v1_grad_want = [
        [3, 4]
    ]
    assert (v1_grad == v1_grad_want).all()
    v2_grad = context.gradient("mul", "var2")
    v2_grad_want = [
        [[1], [2]]
    ]
    assert (v2_grad == v2_grad_want).all()

def test_matmul():
    """More complex matrix multiplication 2x2 x 2x2 = 1x1."""
    graph = function.FunctionGraph()
    graph.add_function("mul", nn_lib.MatMul).set_inputs(
        graph.add_variable("var1", (2, 2)),
        graph.add_variable("var2", (2, 2)))
    context = function.ExecutionContext(
        graph, {"var1": [[0, 1], [2, 3]], "var2": [[4, 5], [6, 7]]})
    comp_got = context.compute("mul")
    comp_want = [
        [6, 7],
        [26, 31]
    ]
    assert (comp_got == comp_want).all()

    var1_grad_got = context.gradient("mul", "var1")
    var1_grad_want = [
        # (0,:,:,:)
        [
            # (0, 0, :, :)
            [
                # o[0,0] = v1[0,0] * v2[0,0] + v1[0,1] * v2[1,0]
                [
                    4, # dv1[0,0] = v2[0,0]
                    6  # dv1[0,1] = v2[1,0]
                ],
                [
                    0, # dv1[1,0] = 0
                    0  # dv1[1,1] = 0
                ]
            ],
            # (0, 1, :, :)
            [
                # o[0,1] = v1[0,0] * v2[0,1] + v1[0,1] * v2[1,1]
                [
                    5, # dv1[0,0] = v2[0,1]
                    7  # dv1[0,1] = v2[1,1]
                ],
                [
                    0, # dv1[1,0] = 0
                    0  # dv1[1,1] = 0
                ]
            ]
        ],
        # (1,:,:,:)
        [
            # (1, 0, :, :)
            [
                # o[1,0] = v1[1,0] * v2[0,0] + v1[1,1] * v2[1,0]
                [
                    0, # dv1[0,0] = 0
                    0  # dv1[0,1] = 0
                ],
                [
                    4, # dv1[1,0] = v2[0,0]
                    6  # dv1[1,1] = v2[1,0]
                ]
            ],
            # (1, 1, :, :)
            [
                # o[1,1] = v1[1,0] * v2[0,1] + v1[1,1] * v2[1,1]
                [
                    0, # dv1[0,0] = 0
                    0  # dv1[0,1] = 0
                ],
                [
                    5, # dv1[1,0] = v2[0,1]
                    7  # dv1[1,1] = v2[1,1]
                ]
            ]
        ]
    ]
    print("got:")
    print(var1_grad_got)
    assert (var1_grad_got == var1_grad_want).all()

    # var1: [[0, 1], [2, 3]]
    # var2: [[4, 5], [6, 7]]
    var2_grad_got = context.gradient("mul", "var2")
    var2_grad_want = [
        # (0,:,:,:)
        [
            # (0, 0, :, :)
            [
                # o[0,0] = v1[0,0] * v2[0,0] + v1[0,1] * v2[1,0]
                [
                    0, # dv2[0,0] = v1[0,0]
                    0  # dv2[0,1] = 0
                ],
                [
                    1, # dv2[1,0] = v1[0,1]
                    0  # dv2[1,1] = 0
                ]
            ],
            # (0, 1, :, :)
            [
                # o[0,1] = v1[0,0] * v2[0,1] + v1[0,1] * v2[1,1]
                [
                    0, # dv2[0,0] = 0
                    0, # dv2[0,1] = v1[0,0]
                ],
                [
                    0, # dv2[1,0] = 0
                    1  # dv2[1,1] = v1[0,1]
                ]
            ]
        ],
        # (1,:,:,:)
        [
            # (1, 0, :, :)
            [
                # o[1,0] = v1[1,0] * v2[0,0] + v1[1,1] * v2[1,0]
                [
                    2, # dv2[0,0] = v1[1,0]
                    0  # dv2[0,1] = 0
                ],
                [
                    3, # dv2[1,0] = v1[1,1]
                    0  # dv2[1,1] = 0
                ]
            ],
            # (1, 1, :, :)
            [
                # o[1,1] = v1[1,0] * v2[0,1] + v1[1,1] * v2[1,1]
                [
                    0, # dv2[0,0] = 0
                    2  # dv2[0,1] = v1[1,0]
                ],
                [
                    0, # dv2[1,0] = 0
                    3  # dv2[1,1] = v1[1,1]
                ]
            ]
        ]
    ]
    assert (var2_grad_got == var2_grad_want).all()
