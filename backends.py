import sys

__all__ = ['get_backend']
__author__ = 'Alex Rogozhnikov'

_backends = {}


def get_backend(tensor):
    for framework_name, (tensor_types, backend) in _backends.items():
        if isinstance(tensor, tensor_types):
            return backend

    for BackendSubclass in AbstractBackend.__subclasses__():
        print('Testing subclass ', BackendSubclass)
        if BackendSubclass.framework_name in sys.modules:
            if BackendSubclass.framework_name not in _backends:
                print('imported ', BackendSubclass.framework_name)
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend.tensor_types(), backend
                if isinstance(tensor, backend.tensor_types()):
                    return backend

    raise RuntimeError('Tensor type unknown')


class AbstractBackend:
    framework_name = None

    def tensor_types(self):
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

    def tensor_types(self):
        return self.np.ndarray,

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


class MXNetNdarrayBackend(AbstractBackend):
    framework_name = 'mxnet.ndarray'

    def __init__(self):
        import mxnet
        self.mx = mxnet

    def tensor_types(self):
        return self.mx.nd.NDArray,

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


class TorchBackend(AbstractBackend):
    framework_name = 'torch'

    def __init__(self):
        import torch
        self.torch = torch

    def tensor_types(self):
        return self.torch.Tensor

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

    def tensor_types(self):
        return self.cupy.ndarray,

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

    def tensor_types(self):
        return self.chainer.Variable

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

    def tensor_types(self):
        return self.tf.Tensor, self.tf.Variable

    def from_numpy(self, x):
        return self.tf.contrib.eager.Variable(x)

    def to_numpy(self, x):
        return x.numpy()

    def arange(self, start, stop):
        return self.tf.range(start, stop)

    def shape(self, x):
        if self.tf.executing_eagerly():
            return x.shape
        else:
            return self.tf.unstack(self.tf.shape(x))

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
        self.K = keras.backend

    def tensor_types(self):
        # здесь методы проверки на самом деле
        raise NotImplementedError

    def from_numpy(self, x):
        return self.K.variable(x, name='einkeras-test')

    def to_numpy(self, x):
        raise NotImplementedError()

    def arange(self, start, stop):
        return self.K.arange(start, stop)

    def shape(self, x):
        return self.K.shape(x)

    def reduce(self, x, operation, axes):
        return getattr(self.K, operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.K.reshape(x, shape)

    def transpose(self, x, axes):
        return self.K.permute_dimensions(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.K.stack(tensors)

    def is_float_type(self, x):
        return 'float'in self.K.dtype(x)

# this one is for static tensorflow
# def tf_wrap_and_compute(function):
#     def returned(x, *args, **kargs):
#         x_placeholder = tf.placeholder(dtype=x.dtype)
#         return tf.Session().run([function(x_placeholder, *args, **kargs)], {x_placeholder: x})[0]
#
#     return returned
