from . import collect_test_backends
from einops.einops import _compatify_pattern_for_einsum
import numpy as np


class Arguments:
    def __init__(self, *args, **kargs):
        self.args = args
        self.kwargs = kargs

    def __call__(self, function):
        return function(*self.args, **self.kwargs)


test_layer_cases = [
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


# Each of the form:
# (Arguments, true_einsum_pattern, in_shapes, out_shape)
test_functional_cases = [
    (
        # Basic:
        Arguments("b c h w, b w -> b h"),
        "abcd,ad->ac",
        ((2, 3, 4, 5), (2, 5)),
        (2, 4),
    ),
    (
        # Three tensors:
        Arguments("b c h w, b w, b c -> b h"),
        "abcd,ad,ab->ac",
        ((2, 3, 40, 5), (2, 5), (2, 3)),
        (2, 40),
    ),
    (
        # Ellipsis, and full names:
        Arguments("... one two three, three four five -> ... two five"),
        "...abc,cde->...be",
        ((32, 5, 2, 3, 4), (4, 5, 6)),
        (32, 5, 3, 6),
    ),
    (
        # Ellipsis at the end:
        Arguments("one two three ..., three four five -> two five ..."),
        "abc...,cde->be...",
        ((2, 3, 4, 32, 5), (4, 5, 6)),
        (3, 6, 32, 5),
    ),
    (
        # Ellipsis on multiple tensors:
        Arguments("... one two three, ... three four five -> ... two five"),
        "...abc,...cde->...be",
        ((32, 5, 2, 3, 4), (32, 5, 4, 5, 6)),
        (32, 5, 3, 6),
    ),
    (
        # One tensor, and underscores:
        Arguments("first_tensor second_tensor -> first_tensor"),
        "ab->a",
        ((5, 4),),
        (5,),
    ),
    (
        # Trace (repeated index)
        Arguments("i i -> "),
        "aa->",
        ((5, 5),),
        (),
    ),
    (
        # Trace with other indices
        Arguments("i middle i -> middle"),
        "aba->b",
        ((5, 10, 5),),
        (10,),
    ),
]


def test_layer():
    for backend in collect_test_backends(layers=True, symbolic=False):
        if backend.framework_name in ['tensorflow', 'torch', 'chainer', 'oneflow']:
            layer_type = backend.layers().EinMix
            for args, in_shape, out_shape in test_layer_cases:
                layer = args(layer_type)
                print('Running', layer.einsum_pattern, 'for', backend.framework_name)
                input = np.random.uniform(size=in_shape).astype('float32')
                input_framework = backend.from_numpy(input)
                output_framework = layer(input_framework)
                output = backend.to_numpy(output_framework)
                assert output.shape == out_shape


def test_functional():
    # Functional tests:
    for backend in collect_test_backends():
        if backend.framework_name in ['tensorflow', 'torch', 'jax', 'numpy']:
            for args, true_pattern, in_shapes, out_shape in test_functional_cases:
                print(f"Running '{args.args[0]}' for {backend.framework_name}")
                predicted_pattern = args(_compatify_pattern_for_einsum)
                assert predicted_pattern == true_pattern
                in_arrays = [
                    np.random.uniform(size=shape).astype('float32')
                    for shape in in_shapes
                ]
                in_arrays_framework = [
                    backend.from_numpy(array)
                    for array in in_arrays
                ]
                out_array = backend.einsum(predicted_pattern, *in_arrays_framework)
                if out_array.shape != out_shape:
                    raise ValueError(
                        f"Expected output shape {out_shape} but got {out_array.shape}"
                    )
                # assert out_array.shape == out_shape


# mxnet/gluon do not support einsum without changing to numpy. which doesn't work with the rest
# in future, after gluon migrated to a new codebase, all testing code will be moved to a new setup
# def test_gluon():
#     for backend in collect_test_backends(layers=True, symbolic=False):
#         if backend.framework_name == 'mxnet.ndarray':
#             import mxnet as mx
#
#             mx.npx.set_np()
#             layer_type = backend.layers().EinMix
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
