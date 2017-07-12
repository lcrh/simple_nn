"""Linear regression via gradient descent on noisy data."""
import matplotlib.pyplot as plot
import numpy as np

import learning
import function
import nn_lib

def build_graph(width):
    """Create linear function + squared diff loss."""
    graph = function.FunctionGraph()
    graph.add_constant("w", (1, width))  # weights
    graph.add_variable("x", (width, 1))  # input
    graph.add_function("out", nn_lib.MatMul).set_inputs("w", "x")

    graph.add_variable("target", (1, 1))  # target
    graph.add_function("diff", nn_lib.Subtract).set_inputs("target", "out")
    graph.add_function("squared_diff",
                       nn_lib.Multiply).set_inputs("diff", "diff")
    return graph

def main():
    """Train linear regression via gradient descent."""
    # Linear regression with squared error.
    graph = build_graph(2)

    training_data = [
        {"x": np.array([[1, i]]),
         "target": np.array([[-3 * i + 5]]) + 10 * np.random.normal(size=(1, 1))}
        for i in xrange(-30, 30)]

    trainer = learning.TrainingContext(
        graph, "squared_diff",
        value_fun=lambda shape: np.random.normal(size=shape))

    def avg_loss():
        sum_loss = 0.0
        for example in training_data:
            trainer.set_variables(example)
            sum_loss += trainer.exec_context.compute("squared_diff")[0, 0]
        return sum_loss / len(training_data)

    for i in xrange(100):
        print("Epoch %d (avg_loss %g)" % (i, avg_loss()))
        trainer.train(training_data, learning.ConstantRate(0.001))

    print("Average loss: %g" % (avg_loss()))
    plot_data_x = [example["x"][0, 1] for example in training_data]
    plot_data_y = [example["target"][0, 0] for example in training_data]

    plot_target_x = range(-30, 30)
    plot_target_y = []
    for i in xrange(-30, 30):
        trainer.set_variables({
            "x": [[1, i]], "target": [[0]]})
        plot_target_y.append(trainer.exec_context.compute("out")[0, 0])

    plot.plot(plot_data_x, plot_data_y)
    plot.plot(plot_target_x, plot_target_y)
    plot.show()

if __name__ == "__main__":
    main()
