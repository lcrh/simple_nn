"""Tests for nn_lib.py."""

import function
import nn_lib

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
