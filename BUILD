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
