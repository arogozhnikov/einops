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
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError()

    def to_numpy(self, x):
        raise NotImplementedError()

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
        return self.mx.nd.array(x)

    def to_numpy(self, x):
        return self.mx.nd.NDArray.asnumpy(x)

    def arange(self, start, stop):
        return self.mx.nd.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.mx.nd.stack(*tensors)

    def is_float_type(self, x):
        return 'float' in str(x.dtype)


# class MXNetBackend(AbstractBackend):
class MXNetBackend:
    framework_name = 'mxnet.symbol'

    def __init__(self):
        import mxnet
        self.mx = mxnet

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.mx.symbol.Symbol)

    def from_numpy(self, x):
        self.last_var_ = self.mx.symbol.Variable('input', shape=x.shape)
        self.last_val_ = x.shape
        return self.last_var_

    def to_numpy(self, x):
        ex = self.last_var_.bind(ctx=self.mx.cpu(), args={'input': self.last_val_})
        ex.forward()
        return ex.outputs[0].asnumpy()

    def shape(self, x):
        return x.infer_shape_partial()[1][0]

    def arange(self, start, stop):
        return self.mx.symbol.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.mx.symbol.stack(*tensors)

    def is_float_type(self, x):
        return 'float' in str(x.infer_type()[1][0])


class TorchBackend(AbstractBackend):
    framework_name = 'torch'

    def __init__(self):
        import torch
        self.torch = torch

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def from_numpy(self, x):
        return self.torch.from_numpy(x)

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


class TensorflowBackend(AbstractBackend):
    framework_name = 'tensorflow'

    def __init__(self):
        import tensorflow
        self.tf = tensorflow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.tf.Tensor, self.tf.Variable))

    def from_numpy(self, x):
        if self.tf.executing_eagerly():
            return self.tf.contrib.eager.Variable(x)
        else:
            return self.tf.placeholder_with_default(x, shape=x.shape, name='einops_placeholder')

    def to_numpy(self, x):
        if self.tf.executing_eagerly():
            return x.numpy()
        else:
            sess = self.tf.Session()
            return sess.run(x)

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


# class KerasBackend(AbstractBackend):
class KerasBackend:
    framework_name = 'keras'

    def __init__(self):
        import keras
        self.keras = keras
        self.K = keras.backend

    def is_appropriate_type(self, tensor):
        return self.K.is_keras_tensor(tensor)

    def from_numpy(self, x):
        self._lastvar = self.keras.Input(batch_shape=x.shape)
        self._lastval = x
        return self._lastvar

    def to_numpy(self, x):
        model = self.keras.models.Model(self._lastvar, x)
        return model.predict_on_batch(self._lastval)

    def arange(self, start, stop):
        return self.K.arange(start, stop)

    def shape(self, x):
        shape = self.K.shape(x)
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
