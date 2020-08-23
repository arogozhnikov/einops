import sys
from doctest import testmod

import numpy

import einops
import einops.layers
import einops.parsing
from einops._backends import AbstractBackend
from einops.einops import rearrange, parse_shape, _optimize_transformation
from . import collect_test_backends

__author__ = 'Alex Rogozhnikov'


def test_doctests_examples():
    if sys.version_info >= (3, 6):
        # python 3.5 and lower do not keep ordered dictionaries
        testmod(einops.layers, raise_on_error=True, extraglobs=dict(np=numpy))
        testmod(einops.einops, raise_on_error=True, extraglobs=dict(np=numpy))


def test_backends_installed():
    """
    This test will fail if some of backends are not installed or can't be imported
    Other tests will just work and only test installed backends.
    """
    from . import skip_cupy
    errors = []
    for backend_type in AbstractBackend.__subclasses__():
        if skip_cupy and backend_type.framework_name == 'cupy':
            continue
        try:
            # instantiate
            backend_type()
        except Exception as e:
            errors.append(e)
    assert len(errors) == 0, errors


def test_optimize_transformations_numpy():
    print('Testing optimizations')
    shapes = [[2] * n_dimensions for n_dimensions in range(14)]
    shapes += [[3] * n_dimensions for n_dimensions in range(6)]
    shapes += [[2, 3, 5, 7]]
    shapes += [[2, 3, 5, 7, 11, 17]]

    for shape in shapes:
        for attempt in range(5):
            n_dimensions = len(shape)
            x = numpy.random.randint(0, 2 ** 12, size=shape).reshape([-1])
            init_shape = shape[:]
            n_reduced = numpy.random.randint(0, n_dimensions + 1)
            reduced_axes = tuple(numpy.random.permutation(n_dimensions)[:n_reduced])
            axes_reordering = numpy.random.permutation(n_dimensions - n_reduced)
            final_shape = numpy.random.randint(0, 1024, size=333)  # just random

            init_shape2, reduced_axes2, axes_reordering2, final_shape2 = combination2 = \
                _optimize_transformation(init_shape, reduced_axes, axes_reordering, final_shape)

            assert numpy.array_equal(final_shape, final_shape2)
            result1 = x.reshape(init_shape).sum(axis=reduced_axes).transpose(axes_reordering).reshape([-1])
            result2 = x.reshape(init_shape2).sum(axis=reduced_axes2).transpose(axes_reordering2).reshape([-1])
            assert numpy.array_equal(result1, result2)

            # testing we can't optimize this formula again
            combination3 = _optimize_transformation(*combination2)
            for a, b in zip(combination2, combination3):
                assert numpy.array_equal(a, b)


def test_parse_shape_imperative():
    backends = collect_test_backends(symbolic=False, layers=False)
    backends += collect_test_backends(symbolic=False, layers=True)
    for backend in backends:
        print('Shape parsing for ', backend.framework_name)
        x = numpy.zeros([10, 20, 30, 40])
        parsed1 = parse_shape(x, 'a b c d')
        parsed2 = parse_shape(backend.from_numpy(x), 'a b c d')
        assert parsed1 == parsed2 == dict(a=10, b=20, c=30, d=40)
        assert parsed1 != dict(a=1, b=20, c=30, d=40) != parsed2

        parsed1 = parse_shape(x, '_ _ _ _')
        parsed2 = parse_shape(backend.from_numpy(x), '_ _ _ _')
        assert parsed1 == parsed2 == dict()

        parsed1 = parse_shape(x, '_ _ _ hello')
        parsed2 = parse_shape(backend.from_numpy(x), '_ _ _ hello')
        assert parsed1 == parsed2 == dict(hello=40)

        parsed1 = parse_shape(x, '_ _ a1 a1a111a')
        parsed2 = parse_shape(backend.from_numpy(x), '_ _ a1 a1a111a')
        assert parsed1 == parsed2 == dict(a1=30, a1a111a=40)


def test_parse_shape_symbolic():
    backends = collect_test_backends(symbolic=True, layers=False)
    backends += collect_test_backends(symbolic=True, layers=True)
    for backend in backends:
        if backend.framework_name == 'keras':
            # need special way to compile, shape vars can be used only inside layers
            continue
        print('special shape parsing for', backend.framework_name)
        input_symbols = [
            backend.create_symbol([10, 20, 30, 40]),
            backend.create_symbol([10, 20, None, None]),
            backend.create_symbol([None, None, None, None]),
        ]
        if backend.framework_name in ['mxnet.symbol']:
            # mxnet can't normally run inference
            input_symbols = [backend.create_symbol([10, 20, 30, 40])]

        for input_symbol in input_symbols:
            shape_placeholder = parse_shape(input_symbol, 'a b c d')
            shape = {}
            for name, symbol in shape_placeholder.items():
                shape[name] = symbol if isinstance(symbol, int) \
                    else backend.eval_symbol(symbol, [(input_symbol, numpy.zeros([10, 20, 30, 40]))])
            print(shape)
            result_placeholder = rearrange(input_symbol, 'a b (c1 c2) (d1 d2) -> (a b d1) c1 (c2 d2)',
                                           **parse_shape(input_symbol, 'a b c1 _'), d2=2)
            result = backend.eval_symbol(result_placeholder, [(input_symbol, numpy.zeros([10, 20, 30, 40]))])
            print(result.shape)
            assert result.shape == (10 * 20 * 20, 30, 1 * 2)
            assert numpy.allclose(result, 0)


def test_is_float_type():
    backends = collect_test_backends(symbolic=False, layers=False)
    backends += collect_test_backends(symbolic=False, layers=True)
    for backend in backends:
        for dtype in ['int32', 'int64', 'float32', 'float64']:
            is_float = 'float' in dtype
            input = numpy.zeros([3, 4, 5], dtype=dtype)
            input = backend.from_numpy(input)
            if 'chainer' in backend.framework_name and not is_float:
                continue  # chainer doesn't allow non-floating tensors
            assert backend.is_float_type(input) == is_float, (dtype, backend, input.dtype)
