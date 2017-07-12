"""Simple functional abstraction for neural networks."""

import abc
import exceptions

import numpy as np

class Error(exceptions.Exception):
    """Base class for function errors."""
    pass

class Function(object):
    """Interface for function objects. Function state should be immutable after
    construction."""
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._graph = None
        self._shape = None

    @abc.abstractmethod
    def execute(self, input_array):
        """Computes function output numpy array based on indexed numpy arrays.

        Returns:
           result array
        """
        return np.array([])

    @abc.abstractmethod
    def gradient(self, index, input_array):
        """Computes gradient for input index, holding other inputs constant.
        Must return a tensor of shape self.shape x input_array[index].shape.
        """
        return np.array([])

    @property
    def graph(self):
        """Returns the graph associated with this function or raises error."""
        if not self._graph:
            raise Error("Function not registered in graph.")
        return self._graph

    @property
    def identifier(self):
        """Returns the identifier of this function."""
        return self.graph.identifier(self)

    def set_input(self, index, input_fun):
        """Assign an input function to this function and returns self."""
        self.graph.set_input(self, index, input_fun)
        return self

    def set_inputs(self, *input_funs):
        """Set all inputs for this function in sequence and returns self."""
        for i, fun in enumerate(input_funs):
            assert fun
            self.set_input(i, fun)
        return self

    def inputs(self):
        """Yield inputs of this function."""
        return self.graph.inputs(self)

    def outputs(self):
        """Yield outputs of this function."""
        return self.graph.outputs(self)

    def indexed_inputs(self):
        """Yields indexed inputs of this function."""
        return self.graph.indexed_inputs(self)

    @abc.abstractmethod
    def get_shape(self, input_shapes):
        """This is called by the executor before the shape property is consulted
        to allow the function to determine its shape dynamically."""
        pass

    @property
    def shape(self):
        """Returns shape. The get_shape function is always called before
        this property is accessed."""
        assert self._shape
        return self._shape

    @shape.setter
    def shape(self, shape):
        assert(shape)
        self._shape = shape

    @property
    def size(self):
        """Returns number of elements in output."""
        return np.prod(self.shape)

    def register_in_graph(self, graph):
        """The FunctionGraph calls this method when adding the function."""
        if self._graph:
            raise Error("Function already registered in another graph.")
        self._graph = graph

class Constant(Function):
    """A constant is a vector that passes through it's input parameter which
    must be assigned in the execution context."""

    def __init__(self, shape):
        """Initialize constant with provided shape."""
        super(Constant, self).__init__()
        assert isinstance(shape, tuple)
        self._var_shape = shape

    def get_shape(self, input_shapes):
        return self._var_shape

    def execute(self, input_array):
        if not input_array[0]:
            raise Error("Missing input.")
        if len(input_array) > 1:
            raise Error("Wrong number of inputs.")
        if input_array[0].shape != self._shape:
            raise Error("Wrong input shape.")
        return input_array[0]

    def gradient(self, index, input_array):
        return np.ones(self.shape)

class Variable(Constant):
    """A variable is a constant that must be explicitly assigned a value."""
    pass

class PointwiseFunction(Function):
    """A function that applies some operation pointwise, i.e., inputs and
    output have the same shape and each input element only influences its
    corresponding output_element."""

    def gradient(self, index, input_array):
        """Uses pointwise gradient internally."""
        # gradient as a matrix.
        grad_matrix = np.zeros((self.size, self.size))
        pt_grad = self.pointwise_gradient(index, input_array).reshape(self.size)
        indices = np.arange(self.size)
        grad_matrix[[indices, indices]] = pt_grad
        return grad_matrix.reshape(self.shape + self.shape)

    @abc.abstractmethod
    def pointwise_gradient(self, index, input_array):
        """Computes pointwise gradient of shape self.shape such that grad[x,y]
        reflects delta fun[x,y] / delta input_array[index][x,y]. All other
        gradients are assumed to be zero."""
        return np.array([])

    def get_shape(self, input_shapes):
        assert input_shapes
        assert all(shape == input_shapes[0] for shape in input_shapes[1:])
        return input_shapes[0]

class FunctionBuilder(object):
    """Helper base class to allow constructing more complex function
    hierarchies."""

    @classmethod
    def instantiate(cls, graph, identifier, *args, **kwargs):
        """Calls constructor (must be provided by subclasses)."""
        return cls(graph, identifier, *args, **kwargs)

    @abc.abstractproperty
    def input_map(self):
        """Return a map or list resolving virtual input indices to actual pairs
        of (function, input_index)."""
        return []

    @abc.abstractproperty
    def output(self):
        """Return the top-level function whose output is the global output."""
        pass

    def set_input(self, index, input_fun):
        """Set interface input by index."""
        resolved_fun, resolved_index = self.input_map[index]
        resolved_fun.set_input(resolved_index, input_fun)
        return self

    def set_inputs(self, *input_funs):
        """Set interface inputs by list."""
        for i, fun in enumerate(input_funs):
            self.set_input(i, fun)
        return self

class FunctionGraph(object):
    """Represents a DAG of connected functions. A directed edge from function
    a to b means that the output of a is an input to b."""

    def __init__(self):
        self._funs = set()
        self._indexed_inputs = {}
        self._inputs = {}
        self._outputs = {}
        self._ids = {}
        self._rev_ids = {}
        self._id_counter = 0

    def add_function(self, identifier, function_type,
                     *constructor_args, **constructor_kw_args):
        """Add a function or function builder to the graph and return it."""
        if not identifier:
            identifier = "_anon_%d" % self._id_counter
            self._id_counter += 1

        if issubclass(function_type, FunctionBuilder):
            result = function_type.instantiate(
                self, identifier, *constructor_args, **constructor_kw_args)
            try:
                if not self._ids[identifier] == result.output:
                    raise Error(
                        "Identifier %s must refer to output %s of %s" % (
                            identifier, str(result.output), str(result)))
                if not self._rev_ids[result.output] == identifier:
                    raise Error(
                        "Output %s of %s must be called %s" % (
                            str(result.output), str(result), identifier))
            except KeyError:
                raise Error(
                    "Output not called %s or not in graph" % identifier)
            return result
        else:
            assert issubclass(function_type, Function)

        fun = function_type(*constructor_args, **constructor_kw_args)
        if identifier in self._ids:
            raise Error("Identifier already exists: " + identifier)
        self._ids[identifier] = fun
        self._rev_ids[fun] = identifier
        self._funs.add(fun)
        self._indexed_inputs[fun] = []
        self._inputs[fun] = set()
        self._outputs[fun] = set()
        fun.register_in_graph(self)
        return fun

    def identifier(self, fun):
        """Return the identifier of a function."""
        return self._rev_ids[fun]

    def lookup(self, fun):
        """Resolve function identifier to function or passes through input.
        Works with lists."""
        if isinstance(fun, (Function, FunctionBuilder)):
            return fun
        if isinstance(fun, str):
            return self._ids[fun]
        return list(map(self.lookup, fun))

    def set_input(self, fun, i, input_fun):
        """Set input_fun to be the ith input to fun."""
        assert not isinstance(fun, Constant)

        if isinstance(input_fun, FunctionBuilder):
            input_fun = input_fun.output
        fun = self.lookup(fun)
        input_fun = self.lookup(input_fun)
        indexed_inputs = self._indexed_inputs[fun]
        if len(indexed_inputs) <= i:
            # Extend inputs list if too short.
            indexed_inputs.extend([None] * (i + 1 - len(indexed_inputs)))
        if indexed_inputs[i] is not None:
            raise Error("Function already has input at indicated index.")
        indexed_inputs[i] = input_fun
        self._inputs[fun].add(input_fun)
        self._outputs[input_fun].add(fun)
        return self

    def outputs(self, fun):
        """Iterate over outputs of function."""
        return (o for o in self._outputs[fun])

    def inputs(self, fun):
        """Iterate over inputs of function."""
        return (i for i in self._inputs[fun])

    def indexed_inputs(self, fun):
        """Yields indexed inputs of selected function."""
        return enumerate(self._indexed_inputs[fun])

    def traverse_topological(self, forward=True, nodes=None):
        """Yield nodes in topological order."""
        edge_map = self._inputs if forward else self._outputs
        seen = set()
        stack = set()
        funs = nodes or self._funs
        for fun in funs:
            for succ in self._traverse_topological_aux(fun, edge_map,
                                                       seen, stack):
                yield succ

    def _traverse_topological_aux(self, fun, edge_map, seen, stack):
        if fun in stack:
            error_msg = " -> ".join(
                f.identifier for f in stack) + " -> " + fun.identifier
            raise Error("Graph is cyclical: " + error_msg)
        if fun in seen:
            return
        seen.add(fun)
        stack.add(fun)
        for node in edge_map[fun]:
            for subnode in self._traverse_topological_aux(node, edge_map,
                                                          seen, stack):
                yield subnode
        yield fun
        stack.remove(fun)

class ExecutionContext(object):
    """Represents data associated with a single execution. Caches intermediate
    results for gradients and function outputs."""

    def __init__(self, graph, value_map, value_fun=np.zeros):
        """Initialize a new execution context for a FunctionGraph. Variables
        must be explicitly assigned via value_map, constants may also be
        assigned with value_fun. Only variables and constants can be
        assigned."""
        # Cache from function to output as vector.
        self._output = {}
        # Cache from (function, delta_fun) to gradient as m x n matrix.
        self._gradients = {}
        self._graph = graph

        # Map to transitive predecessor.
        self._predec = {}

        # Resolve function identifiers.
        value_map = {graph.lookup(key): val
                     for key, val in value_map.iteritems()}

        # Determine shapes and initialize variables, implicitly ensures that
        # the graph is acyclic.
        for fun in graph.traverse_topological():
            fun = self._graph.lookup(fun)

            if any(p is None for _, p in fun.indexed_inputs()):
                raise Error("Function " + str(fun) + " has unassigned child.")

            # Set shape:
            input_shapes = [p.shape for _, p in fun.indexed_inputs()]
            fun.shape = fun.get_shape(input_shapes)

            # Assign variables
            if isinstance(fun, Constant):
                try:
                    val = np.array(value_map[fun])
                    self._output[fun] = val.reshape(fun.shape)
                except KeyError:
                    if isinstance(fun, Variable):
                        raise Error("Unassigned variable: %s (%s)" % (fun.identifier, fun))
                    self._output[fun] = np.array(value_fun(fun.shape))

            # Set predecessors.
            predecs = {fun}
            for predec in fun.inputs():
                predecs.update(self._predec[predec])
            self._predec[fun] = predecs

    def get_input_array(self, fun):
        """Returns the list of input values for the given function."""
        fun = self._graph.lookup(fun)
        result = []
        for _, predec_fun in fun.indexed_inputs():
            if predec_fun:
                result.append(self.compute(predec_fun))
            else:
                result.append(None)
        return np.array(result)

    def compute(self, fun):
        """Return output of given function. Caches intermediate results internally."""
        fun = self._graph.lookup(fun)
        try:
            return self._output[fun]
        except KeyError:
            pass
        output = fun.execute(self.get_input_array(fun)).reshape(fun.shape)
        self._output[fun] = output
        return output

    def gradient(self, fun, delta_fun):
        """Computes the gradient of fun with regards to changes in delta_fun.
        Output is of type fun.shape x delta_fun.shape.
        """
        fun = self._graph.lookup(fun)
        delta_fun = self._graph.lookup(delta_fun)
        try:
            return self._gradients[(fun, delta_fun)]
        except KeyError:
            pass

        if fun is delta_fun:
            return np.identity(delta_fun.size)

        # Processing as 1D arrays from here:
        expected_shape = (fun.size, delta_fun.size)
        input_array = self.get_input_array(fun)
        result = np.zeros(expected_shape)

        for i, pred in fun.indexed_inputs():
            # Skip if gradient is guaranteed to be zero:
            if not delta_fun in self._predec[pred]:
                continue

            # Chain rule:

            # fun.size x pred.size:
            fun_grad = fun.gradient(i, input_array).reshape(fun.size, pred.size)

            # pred.size x delta_fun.size:
            pred_grad = self.gradient(pred, delta_fun).reshape(pred.size,
                                                               delta_fun.size)

            # fun.size x delta_fun.size
            dot_prod = fun_grad.dot(pred_grad)
            result += dot_prod
        self._gradients[(fun, delta_fun)] = result.reshape(
            fun.shape + delta_fun.shape)
        shaped = result.reshape(fun.shape + delta_fun.shape)
        return shaped
