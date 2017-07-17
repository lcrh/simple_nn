"""Learn to classify a non-linear example using a multilayer net."""

import random

import matplotlib.pyplot as plot
import numpy as np

import learning
import function
import nn_lib

def build_graph(input_width, layer_sizes, activation_function):
    """Create deep neural classifier with cross-entropy loss."""
    graph = function.FunctionGraph()
    inp = graph.add_variable("x", (1, input_width))  # input
    for i, layer_size in enumerate(layer_sizes):
        inp = graph.add_function("layer_%d" % i,
                                 nn_lib.FullyConnected, 1,
                                 input_width, layer_size,
                                 activation_function).set_inputs(inp)
        input_width = layer_size

    # Compute output.
    graph.add_function(
        "output", nn_lib.FullyConnected, 1,
        input_width, 1, nn_lib.Sigmoid).set_inputs(inp)

    graph.add_variable("target", (1, 1))  # target

    # Create loss function.
    graph.add_function("loss", nn_lib.CrossEntropy).set_inputs(
        "output", "target")

    return graph

def gen_training_data():
    """Create training data."""
    max_n_examples = 400
    training_data = []

    for _ in xrange(max_n_examples):
        x, y = (np.random.random(2) - 0.5) * 2
        if not filter_examples(x, y):
            continue
        classification = gen_classification(x, y)

        training_data.append({
            "x": np.array([[x, y]]),
            "target": np.array([[1.0 if classification else 0.0]])
        })
    return training_data

def gen_classification(x, y):
    """Create boolean target tag."""
    return x * y > 0

def filter_examples(x, y):
    """Return true if random example should be kept."""
    margin = 0.1
    return not (
        -margin <= x <= margin or
        -margin <= y <= margin)

def batches(total, batch_size):
    """Yield total as batches of batch_size."""
    i = 0
    while i < len(total):
        yield total[i : i + batch_size]
        i += batch_size

def random_batches(total, batch_size):
    """Yield total as shuffled batches of batch_size."""
    shuffled = list(total)
    random.shuffle(shuffled)
    return batches(shuffled, batch_size)

def main():
    """Train deep classifier."""
    graph = build_graph(2, [8, 4], nn_lib.ReLU)

    training_data = gen_training_data()

    trainer = learning.TrainingContext(
        graph, "loss",
        value_fun=lambda shape: 2 * (1 - 2 * np.random.random(size=shape)))

    def avg_loss():
        """Compute average loss."""
        sum_loss = 0.0
        for example in training_data:
            trainer.set_variables(example)
            loss = trainer.exec_context.compute("loss")[0, 0]
            sum_loss += loss

        return sum_loss / len(training_data)

    batch_size = 1
    epochs = 30
    init_rate = 0.15
    decay_param = 1.5
    for i in xrange(epochs):
        rate = init_rate / (1 + i * decay_param)
        print("Epoch %d (avg_loss %g, alpha %g)" % (i, avg_loss(), rate))
        for batch in random_batches(training_data, batch_size):
            learner = learning.ConstantRate(rate)
            trainer.train(batch, learner)

    print("Average loss: %g" % (avg_loss()))

    print("Parameter values:")
    for fun, val in trainer.parameters.iteritems():
        print("== %s:\n%s" % (fun.identifier, val))

    plot_data_x = np.array([example["x"][0, 0] for example in training_data])
    plot_data_y = np.array([example["x"][0, 1] for example in training_data])
    classification = gen_classification(plot_data_x, plot_data_y)
    color = np.where(classification, "r", "b")

    grid_resolution = 32
    preds = np.zeros((grid_resolution, grid_resolution))
    for grid_x in xrange(grid_resolution):
        for grid_y in xrange(grid_resolution):
            x = (grid_x + 0.5) * 2.0 / grid_resolution - 1
            y = (grid_y + 0.5) * 2.0 / grid_resolution - 1

            trainer.set_variables({
                "x": [[x, y]],
                "target": [[0]]})
            val = trainer.exec_context.compute("output")[0, 0]
            preds[grid_x, grid_y] = val

    plot.imshow(preds.transpose(), extent=[-1, 1, -1, 1], interpolation="bicubic")
    plot.scatter(plot_data_x, plot_data_y, c=color)
    plot.show()

if __name__ == "__main__":
    main()
