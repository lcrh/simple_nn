python_library(
    name = "function",
    sources = [
        "function.py",
    ],
    dependencies = [
        "3rdparty/python:numpy",
    ],
)

python_tests(
    name = "function_test",
    sources = ["function_test.py"],
    dependencies = [
        ":function",
        "3rdparty/python:numpy",
    ],
)

python_library(
    name = "nn_lib",
    sources = [
        "nn_lib.py",
    ],
    dependencies = [
        ":function",
        "3rdparty/python:numpy",
    ],
)

python_tests(
    name = "nn_lib_test",
    sources = [
        "nn_lib_test.py",
    ],
    dependencies = [
        ":function",
        ":nn_lib",
    ],
)
