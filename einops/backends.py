import sys

__all__ = ['get_backend']
__author__ = 'Alex Rogozhnikov'

_backends = {}
_debugging = False


def get_backend(tensor):
    for framework_name, backend in _backends.items():
        if backend.is_appropriate_type(tensor):
            return backend

    for BackendSubclass in AbstractBackend.__subclasses__():
        if _debugging:
            print('Testing for subclass of ', BackendSubclass)
        if BackendSubclass.framework_name not in _backends:
            if BackendSubclass.framework_name in sys.modules:
                if _debugging:
                    print('Imported backend for ', BackendSubclass.framework_name)
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    return backend

    raise RuntimeError('Tensor type unknown to einops {}'.format(type(tensor)))


class AbstractBackend:
    """ Base backend class, major part of methods are only for debugging purposes. """
    framework_name = None

    def is_appropriate_type(self, tensor):
        """ helper method should recognize tensors it can handle """
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        raise NotImplementedError()

    def shape(self, x):
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def is_float_type(self, x):
        raise NotImplementedError()

    def __repr__(self):
        return "<einops backend for {}>".format(self.framework_name)


class UnknownSize:
    """ pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements """

    def __floordiv__(self, other):
        return self

    def __eq__(self, other):
        return True  # we don't know actual size

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __hash__(self):
        return None.__hash__()


class NumpyBackend(AbstractBackend):
    framework_name = 'numpy'

    def __init__(self):
        import numpy
        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return x

    def to_numpy(self, x):
        return x

    def arange(self, start, stop):
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.np.stack(tensors)

    def is_float_type(self, x):
        return x.dtype in ('float16', 'float32', 'float64', 'float128')


class GluonBackend(AbstractBackend):
    framework_name = 'mxnet.ndarray'

    def __init__(self):
        import mxnet
        self.mx = mxnet

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.mx.nd.NDArray)

    def from_numpy(self, x):
        var = self.mx.nd.array(x)
        var.attach_grad()
        return var

    def to_numpy(self, x):
        return self.mx.nd.NDArray.asnumpy(x)

    def reshape(self, x, shape):
        if len(shape) == 0:
            return x  # poor support of scalars in mxnet
        return x.reshape(shape)

    def arange(self, start, stop):
        return self.mx.nd.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.mx.nd.stack(*tensors)

    def is_float_type(self, x):
        return 'float' in str(x.dtype)

    def layers(self):
        from .layers import gluon
        return gluon


class MXNetBackend(AbstractBackend):
    framework_name = 'mxnet.symbol'

    def __init__(self):
        import mxnet
        self.mx = mxnet

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.mx.symbol.Symbol)

    def create_symbol(self, shape, dtype='float32'):
        shape = tuple(0 if d is None else d for d in shape)
        var = self.mx.symbol.Variable('input', shape=shape, dtype=dtype)
        return var

    def eval_symbol(self, symbol, input_dict):
        args = {var.name: self.mx.nd.array(val) for var, val in input_dict}
        ex = symbol.bind(ctx=self.mx.cpu(), args=args)
        ex.forward()
        return ex.outputs[0].asnumpy()

    def shape(self, x):
        # mxnet has problems with shape inference - it does not provide shape variables
        # shape_array seems to be impossible to use in shape inference
        # infer_shape_partial returns empty tuple if was not able to infer shape
        # reductions such as sum can't return scalars, but return 1-element vectors
        shape = x.infer_shape_partial()[1][0]
        shape = tuple(UnknownSize() if d == 0 else d for d in shape)
        return shape

    def reshape(self, x, shape):
        if len(shape) == 0:
            return x  # poor support of scalars in mxnet
        if any(isinstance(dimension, UnknownSize) for dimension in shape):
            from .einops import EinopsError
            raise EinopsError("Mxnet could't infer all dimensions statically, please provide those with axes_lengths")
        return x.reshape(shape)

    def arange(self, start, stop):
        return self.mx.symbol.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.mx.symbol.stack(*tensors)

    def is_float_type(self, x):
        return 'float' in str(x.infer_type()[1][0])

    def layers(self):
        from .layers import gluon
        return gluon


class TorchBackend(AbstractBackend):
    framework_name = 'torch'

    def __init__(self):
        import torch
        self.torch = torch

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def from_numpy(self, x):
        variable = self.torch.from_numpy(x)
        variable.requires_grad = True
        return variable

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop):
        return self.torch.arange(start, stop, dtype=self.torch.int64)

    def reduce(self, x, operation, reduced_axes):
        for axis in sorted(reduced_axes, reverse=True):
            if operation == 'min':
                x, _ = x.min(dim=axis)
            elif operation == 'max':
                x, _ = x.max(dim=axis)
            elif operation in ['sum', 'mean', 'prod']:
                x = getattr(x, operation)(dim=axis)
            else:
                raise NotImplementedError('Unknown reduction ', operation)
        return x

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.torch.stack(tensors)

    def is_float_type(self, x):
        return x.dtype in [self.torch.float16, self.torch.float32, self.torch.float64]

    def layers(self):
        from .layers import torch
        return torch


class CupyBackend(AbstractBackend):
    framework_name = 'cupy'

    def __init__(self):
        import cupy
        self.cupy = cupy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.cupy.ndarray)

    def from_numpy(self, x):
        return self.cupy.asarray(x)

    def to_numpy(self, x):
        return self.cupy.asnumpy(x)

    def arange(self, start, stop):
        return self.cupy.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.cupy.stack(tensors)

    def is_float_type(self, x):
        return x.dtype in ('float16', 'float32', 'float64', 'float128')


class ChainerBackend(AbstractBackend):
    framework_name = 'chainer'

    def __init__(self):
        import chainer
        import cupy
        self.chainer = chainer
        self.cupy = cupy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.chainer.Variable)

    def from_numpy(self, x):
        return self.chainer.Variable(self.cupy.asarray(x, dtype='float32'))

    def to_numpy(self, x):
        if isinstance(x, self.chainer.Variable):
            x = x.data
        return self.cupy.asnumpy(x)

    def arange(self, start, stop):
        return self.cupy.arange(start, stop)

    def reduce(self, x, operation, axes):
        return getattr(self.chainer.functions, operation)(x, axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.chainer.functions.stack(tensors)

    def is_float_type(self, x):
        return x.dtype in ('float16', 'float32', 'float64', 'float128')

    def layers(self):
        from .layers import chainer
        return chainer


class TensorflowBackend(AbstractBackend):
    framework_name = 'tensorflow'

    def __init__(self):
        import tensorflow
        self.tf = tensorflow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.tf.Tensor, self.tf.Variable))

    def from_numpy(self, x):
        assert self.tf.executing_eagerly()
        return self.tf.contrib.eager.Variable(x)

    def to_numpy(self, x):
        assert self.tf.executing_eagerly()
        return x.numpy()

    def create_symbol(self, shape, dtype='float32'):
        assert not self.tf.executing_eagerly()
        return self.tf.placeholder(dtype=dtype, shape=shape, name='einops_placeholder')

    def eval_symbol(self, symbol, input_dict):
        assert not self.tf.executing_eagerly()
        with self.tf.Session() as sess:
            return sess.run(symbol, feed_dict=dict(input_dict))

    def arange(self, start, stop):
        return self.tf.range(start, stop)

    def shape(self, x):
        if self.tf.executing_eagerly():
            return tuple(int(d) for d in x.shape)
        else:
            return tuple(self.tf.unstack(self.tf.shape(x)))

    def reduce(self, x, operation, axes):
        return getattr(self.tf, 'reduce_' + operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.tf.reshape(x, shape)

    def transpose(self, x, axes):
        return self.tf.transpose(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.tf.stack(tensors)

    def is_float_type(self, x):
        return x.dtype in ('float16', 'float32', 'float64', 'float128')


class KerasBackend(AbstractBackend):
    framework_name = 'keras'

    def __init__(self):
        import keras
        self.keras = keras
        self.K = keras.backend

    def is_appropriate_type(self, tensor):
        return self.K.is_tensor(tensor) and self.K.is_keras_tensor(tensor)

    def create_symbol(self, shape):
        return self.keras.Input(batch_shape=shape)

    def eval_symbol(self, symbol, input_dict):
        (variable, value), = input_dict
        model = self.keras.models.Model(variable, symbol)
        return model.predict_on_batch(value)

    def arange(self, start, stop):
        return self.K.arange(start, stop)

    def shape(self, x):
        shape = self.K.shape(x)  # tf tensor (if tf is backend)
        return tuple(shape[i] for i in range(shape.shape[0]))

    def reduce(self, x, operation, axes):
        return getattr(self.K, operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.K.reshape(x, shape)

    def transpose(self, x, axes):
        return self.K.permute_dimensions(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.K.stack(tensors)

    def is_float_type(self, x):
        return 'float' in self.K.dtype(x)

    def layers(self):
        from .layers import keras
        return keras
