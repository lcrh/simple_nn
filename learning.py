"""Basic interfaces and classes for learning functions."""

import abc
import exceptions

import numpy as np

import function

class Error(exceptions.Exception):
    """Base class for errors."""
    pass

class LearningRule(object):
    """Interface for learning functions."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def register_gradient_map(self, grad_map):
        """Registers a new gradient map with the learning rule."""
        pass

    @abc.abstractmethod
    def reset(self):
        """Reset for a new batch."""
        pass

    @abc.abstractmethod
    def update_map(self, value_map):
        """Receives a map of parameter values and a list of gradient maps
        and returns a map of parameter value updates."""
        pass

class ConstantRate(LearningRule):
    """Sum gradients and multiply by given rate."""
    def __init__(self, rate=0.1):
        self._rate = rate
        self._grad = {}
        self._num = {}

    def register_gradient_map(self, grad_map):
        for fun, val in grad_map.iteritems():
            if fun not in self._grad:
                self._grad[fun] = val
                self._num[fun] = 1
            else:
                self._grad[fun] += val
                self._num[fun] += 1

    def reset(self):
        self._grad = {}
        self._num = {}

    def update_map(self, value_map):
        updates = {}
        for fun, val in self._grad.iteritems():
            new_val = value_map[fun] - (val / self._num[fun]) * self._rate
            updates[fun] = new_val
                
        return updates

class TrainingContext(object):
    """Context object that stores info about a training run."""

    def __init__(self, graph, objective_fun,
                 value_map=None, value_fun=np.zeros,
                 frozen=None):
        """Create a new trainer object. Value_map contains initial value
        assignments for parameters. Frozen parameters are not changed during
        training."""
        self._graph = graph
        self._objective_fun = self._graph.lookup(objective_fun)
        self._var_map = None
        self._value_map = {}
        self._frozen = set(frozen if frozen else [])
        self._exec_context = None
        if value_map:
            self.update_parameters(value_map)

        # Initialize uninitialized parameters.
        for fun in self._graph.traverse_topological():
            if (isinstance(fun, function.Parameter) and
                    not isinstance(fun, function.Variable)):
                if fun not in self._value_map:
                    self._value_map[fun] = value_fun(fun.shape)

    def train(self, var_maps, learning_rule):
        """Perform training with the given learning rule.

        Args:
            var_maps: An list of inputs to be processed in batch.
            learning_rule: A LearningRule object.
            batch_size: How many elements to pass at once to the learning_rule.
        """
        learning_rule.reset()
        for var_map in var_maps:
            learning_rule.register_gradient_map(self.gradient_map(var_map))
        self.update_parameters(learning_rule.update_map(self._value_map))

    def update_parameters(self, value_map):
        """Update parameter values to values indicated in value_map."""
        for fun, val in value_map.iteritems():
            fun = self._graph.lookup(fun)
            if not isinstance(fun, function.Parameter):
                raise Error("Can't update non-parameter function.")
            if isinstance(fun, function.Variable):
                raise Error("Can't update variable.")
            if fun in self._frozen:
                raise Error("Can't update frozen parameter.")
            self._value_map[fun] = val

    @property
    def parameters(self):
        return dict(self._value_map)

    def freeze(self, fun):
        """Freeze a function."""
        self._frozen.add(self._graph.lookup(fun))

    def unfreeze(self, fun):
        """Unfreeze a function."""
        self._frozen.remove(self._graph.lookup(fun))

    def set_variables(self, var_map):
        """Set variable assignments."""
        var_map = {
            self._graph.lookup(fun): val for fun, val in var_map.iteritems()
        }
        if not all(isinstance(fun, function.Variable)
                   for fun in var_map.iterkeys()):
            raise Error("Non-variable in input map.")
        for fun in self._graph.traverse_topological():
            if isinstance(fun, function.Variable):
                if not fun in var_map:
                    raise Error("Missing assignment to variable " + fun.name)
        self._var_map = var_map
        self._exec_context = None

    def objective_value(self, var_map=None):
        """Return value of the objective function as a scalar."""
        if var_map:
            self.set_variables(var_map)
        return self.exec_context.compute(self._objective_fun).reshape(1)[0]

    @property
    def exec_context(self):
        """Returns an execution context corresponding to the current variable
        assignment."""
        if not self._exec_context:
            if not self._var_map:
                raise Error("No input set.")
            value_map = dict(self._value_map)
            for var, val in self._var_map.iteritems():
                value_map[var] = val
            self._exec_context = function.ExecutionContext(
                self._graph, value_map)
        return self._exec_context

    def gradient_map(self, var_map=None):
        """Returns a map from non-frozen parameters c to d/dc."""
        if var_map:
            self.set_variables(var_map)
        result = {}
        for fun in self._graph.traverse_topological():
            if not isinstance(fun, function.Parameter):
                continue
            if isinstance(fun, function.Variable):
                continue
            if fun in self._frozen:
                continue
            result[fun] = self.exec_context.gradient(
                self._objective_fun, fun).reshape(fun.shape)
        return result
