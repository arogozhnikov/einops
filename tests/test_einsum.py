from . import collect_test_backends
from einops.einops import _compactify_pattern_for_einsum
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
        "b c h w, b w -> b h",
        "abcd,ad->ac",
        ((2, 3, 4, 5), (2, 5)),
        (2, 4),
    ),
    (
        # Three tensors:
        "b c h w, b w, b c -> b h",
        "abcd,ad,ab->ac",
        ((2, 3, 40, 5), (2, 5), (2, 3)),
        (2, 40),
    ),
    (
        # Ellipsis, and full names:
        "... one two three, three four five -> ... two five",
        "...abc,cde->...be",
        ((32, 5, 2, 3, 4), (4, 5, 6)),
        (32, 5, 3, 6),
    ),
    (
        # Ellipsis at the end:
        "one two three ..., three four five -> two five ...",
        "abc...,cde->be...",
        ((2, 3, 4, 32, 5), (4, 5, 6)),
        (3, 6, 32, 5),
    ),
    (
        # Ellipsis on multiple tensors:
        "... one two three, ... three four five -> ... two five",
        "...abc,...cde->...be",
        ((32, 5, 2, 3, 4), (32, 5, 4, 5, 6)),
        (32, 5, 3, 6),
    ),
    (
        # One tensor, and underscores:
        "first_tensor second_tensor -> first_tensor",
        "ab->a",
        ((5, 4),),
        (5,),
    ),
    (
        # Trace (repeated index)
        "i i -> ",
        "aa->",
        ((5, 5),),
        (),
    ),
    (
        # Too many spaces in string:
        " one  two  ,  three four->two  four  ",
        "ab,cd->bd",
        ((2, 3), (4, 5)),
        (3, 5),
    ),
    # The following tests were inspired by numpy's einsum tests
    # https://github.com/numpy/numpy/blob/v1.23.0/numpy/core/tests/test_einsum.py
    (
        # Trace with other indices
        "i middle i -> middle",
        "aba->b",
        ((5, 10, 5),),
        (10,),
    ),
    (
        # Ellipsis in the middle:
        "i ... i -> ...",
        "a...a->...",
        ((5, 3, 2, 1, 4, 5),),
        (3, 2, 1, 4),
    ),
    (
        # Product of first and last axes:
        "i ... i -> i ...",
        "a...a->a...",
        ((5, 3, 2, 1, 4, 5),),
        (5, 3, 2, 1, 4),
    ),
    (
        # Triple diagonal
        "one one one -> one",
        "aaa->a",
        ((5, 5, 5),),
        (5,),
    ),
    (
        # Axis swap:
        "i j k -> j i k",
        "abc->bac",
        ((1, 2, 3),),
        (2, 1, 3),
    ),
    (
        # Identity:
        "... -> ...",
        "...->...",
        ((5, 4, 3, 2, 1),),
        (5, 4, 3, 2, 1),
    ),
    (
        # Elementwise product of three tensors
        "..., ..., ... -> ...",
        "...,...,...->...",
        ((3, 2), (3, 2), (3, 2)),
        (3, 2),
    ),
    (
        # Basic summation:
        "index ->",
        "a->",
        ((10,)),
        (()),
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


valid_backends_functional = ['tensorflow', 'torch', 'jax', 'numpy',
                             'chainer', 'oneflow', 'cupy', 'tensorflow.keras']

def test_functional():
    # Functional tests:
    backends = filter(lambda x: x.framework_name in valid_backends_functional,
                      collect_test_backends())
    for backend in backends:
        for einops_pattern, true_pattern, in_shapes, out_shape in test_functional_cases:
            print(f"Running '{einops_pattern}' for {backend.framework_name}")
            
            # Create pattern:
            predicted_pattern = _compactify_pattern_for_einsum(einops_pattern)
            assert predicted_pattern == true_pattern

            # Generate example data:
            rstate = np.random.RandomState(0)
            in_arrays = [
                rstate.uniform(size=shape).astype('float32')
                for shape in in_shapes
            ]
            in_arrays_framework = [
                backend.from_numpy(array) for array in in_arrays
            ]
            
            # Actually run einsum:
            out_array = backend.einsum(predicted_pattern, *in_arrays_framework)
            
            # Check shape:
            if out_array.shape != out_shape:
                raise ValueError(
                    f"Expected output shape {out_shape} but got {out_array.shape}"
                )

            # Check values:
            true_out_array = np.einsum(true_pattern, *in_arrays)
            predicted_out_array = backend.to_numpy(out_array)
            np.testing.assert_array_almost_equal(predicted_out_array,
                                                    true_out_array,
                                                    decimal=5)


def test_functional_symbolic():
    backends = filter(lambda x: x.framework_name in valid_backends_functional,
                      collect_test_backends(symbolic=True, layers=False))
    for backend in backends:
        for einops_pattern, true_pattern, in_shapes, out_shape in test_functional_cases:
            print(f"Running '{einops_pattern}' for symbolic {backend.framework_name}")
            # Create pattern:
            predicted_pattern = _compactify_pattern_for_einsum(einops_pattern)
            assert predicted_pattern == true_pattern

            rstate = np.random.RandomState(0)
            in_syms = [backend.create_symbol(in_shape) for in_shape in in_shapes]
            in_data = [rstate.uniform(size=in_shape).astype('float32') for in_shape in in_shapes]

            expected_out_data = np.einsum(true_pattern, *in_data)
            predicted_out_symbol = backend.einsum(predicted_pattern, *in_syms)
            print(predicted_out_symbol)

            predicted_out_data = backend.eval_symbol(
                predicted_out_symbol,
                list(zip(in_syms, in_data)),
            )
            if predicted_out_data.shape != out_shape:
                raise ValueError(
                    f"Expected output shape {out_shape} but got {predicted_out_data.shape}"
                )
            assert np.testing.assert_array_almost_equal(predicted_out_data,
                                                        expected_out_data,
                                                        decimal=5)


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
