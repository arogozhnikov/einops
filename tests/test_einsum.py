from . import collect_test_backends
import numpy as np


class Arguments:
    def __init__(self, *args, **kargs):
        self.args = args
        self.kwargs = kargs

    def __call__(self, function):
        return function(*self.args, **self.kwargs)


test_cases = [
    (
        Arguments('b c_in h w -> w c_out h b', 'c_in c_out', bias_shape=None, c_out=13, c_in=12),
        (2, 12, 3, 4),
        (4, 13, 3, 2),
    ),
    (
        Arguments('b c_in h w -> w c_out h b', 'c_in c_out', bias_shape='c_out', c_out=13, c_in=12),
        (2, 12, 3, 4),
        (4, 13, 3, 2),
    ),
    (
        Arguments('b c_in h w -> w c_in h b', '', bias_shape=None, c_in=12),
        (2, 12, 3, 4),
        (4, 12, 3, 2),
    ),
    (
        Arguments('b c_in h w -> b c_out', 'c_in h w c_out', bias_shape=None, c_in=12, h=3, w=4, c_out=5),
        (2, 12, 3, 4),
        (2, 5),
    ),
    (
        Arguments('b t head c_in -> b t head c_out', 'head c_in c_out', bias_shape=None, head=4, c_in=5, c_out=6),
        (2, 3, 4, 5),
        (2, 3, 4, 6),
    ),
]


def test_all():
    for backend in collect_test_backends(layers=True, symbolic=False):
        if backend.framework_name in ['tensorflow', 'torch', 'chainer']:
            layer_type = backend.layers().WeightedEinsum
            for args, in_shape, out_shape in test_cases:
                layer = args(layer_type)
                print('Running', layer.einsum_pattern, 'for', backend.framework_name)
                input = np.random.uniform(size=in_shape).astype('float32')
                input_framework = backend.from_numpy(input)
                output_framework = layer(input_framework)
                output = backend.to_numpy(output_framework)
                assert output.shape == out_shape

# mxnet/gluon do not support einsum without changing to numpy. which doesn't work with the rest
# in future, after gluon migrated to a new codebase, all testing code will be moved to a new setup
# def test_gluon():
#     for backend in collect_test_backends(layers=True, symbolic=False):
#         if backend.framework_name == 'mxnet.ndarray':
#             import mxnet as mx
#
#             mx.npx.set_np()
#             layer_type = backend.layers().WeightedEinsum
#
#             for args, in_shape, out_shape in test_cases:
#                 layer = args(layer_type)
#                 # gluon requires initialization
#                 layer.initialize()
#                 input = np.random.uniform(size=in_shape).astype('float32')
#                 input_framework = mx.np.array(input)
#                 output_framework = layer(input_framework)
#                 output = backend.to_numpy(output_framework)
#                 assert output.shape == out_shape
