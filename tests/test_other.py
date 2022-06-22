import sys
import unittest
from doctest import testmod
from typing import Dict, List, Optional

import numpy
from nose.tools import assert_raises
from parameterized import parameterized, parameterized_class

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
    from . import skip_cupy, skip_oneflow, skip_mindspore
    errors = []
    for backend_type in AbstractBackend.__subclasses__():
        if skip_cupy and backend_type.framework_name == 'cupy':
            continue
        if skip_oneflow and backend_type.framework_name == 'oneflow':
            continue
        if skip_mindspore and backend_type.framework_name == 'mindspore':
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


_IMPERATIVE_BACKENDS = [{"backend": backend} for backend in
                        (collect_test_backends(symbolic=False, layers=False) +
                         collect_test_backends(symbolic=False, layers=True))]


@parameterized_class(_IMPERATIVE_BACKENDS)
class TestParseShapeImperative(unittest.TestCase):
    def setUp(self):
        self.x = numpy.zeros([10, 20, 30, 40])

    def test_parse_shape_imperative(self):
        print('Shape parsing for ', self.backend.framework_name)
        parsed1 = parse_shape(self.x, 'a b c d')
        parsed2 = parse_shape(self.backend.from_numpy(self.x), 'a b c d')
        assert parsed1 == parsed2 == dict(a=10, b=20, c=30, d=40)
        assert parsed1 != dict(a=1, b=20, c=30, d=40) != parsed2

    def test_underscore(self):
        parsed1 = parse_shape(self.x, '_ _ _ _')
        parsed2 = parse_shape(self.backend.from_numpy(self.x), '_ _ _ _')
        assert parsed1 == parsed2 == dict()

    def test_underscore_one(self):
        parsed1 = parse_shape(self.x, '_ _ _ hello')
        parsed2 = parse_shape(self.backend.from_numpy(self.x), '_ _ _ hello')
        assert parsed1 == parsed2 == dict(hello=40)

    def test_underscore_several(self):
        parsed1 = parse_shape(self.x, '_ _ a1 a1a111a')
        parsed2 = parse_shape(self.backend.from_numpy(self.x), '_ _ a1 a1a111a')
        assert parsed1 == parsed2 == dict(a1=30, a1a111a=40)

    def test_repeating(self):
        with assert_raises(einops.EinopsError):
            parse_shape(self.x, 'a a b b')

        with assert_raises(einops.EinopsError):
            parse_shape(self.backend.from_numpy(self.x), 'a a b b')

    @parameterized.expand([
        ([10, 20], '...', dict()),
        ([10], '... a', dict(a=10)),
        ([10, 20], '... a', dict(a=20)),
        ([10, 20, 30], '... a', dict(a=30)),
        ([10, 20, 30, 40], '... a', dict(a=40)),
        ([10], 'a ...', dict(a=10)),
        ([10, 20], 'a ...', dict(a=10)),
        ([10, 20, 30], 'a ...', dict(a=10)),
        ([10, 20, 30, 40], 'a ...', dict(a=10)),
        ([10, 20, 30, 40], ' a ... b', dict(a=10, b=40)),
        ([10, 40], ' a ... b', dict(a=10, b=40)),
    ])
    def test_ellipsis(self, shape: List[int], pattern: str,
                      expected: Dict[str, int]):
        x = numpy.ones(shape)
        parsed1 = parse_shape(x, pattern)
        parsed2 = parse_shape(self.backend.from_numpy(x), pattern)
        assert parsed1 == parsed2 == expected


_SYMBOLIC_BACKENDS = [{"backend": backend} for backend in
                      (collect_test_backends(symbolic=True, layers=False) +
                       collect_test_backends(symbolic=True, layers=True))
                      if backend.framework_name != 'tensorflow.keras']
# tensorflow.keras needs special way to compile,
# shape vars can be used only inside layers but not as outputs


@parameterized_class(_SYMBOLIC_BACKENDS)
class TestParseShapeSymbolic(unittest.TestCase):
    @parameterized.expand([
        ([10, 20, 30, 40],),
        ([10, 20, None, None],),
        ([None, None, None, None],),
    ])
    def test_parse_shape_symbolic(self, shape):
        print('special shape parsing for', self.backend.framework_name)
        if self.backend.framework_name in ['mxnet.symbol']:
            # mxnet can't normally run inference
            shape = [10, 20, 30, 40]
        input_symbol = self.backend.create_symbol(shape)

        shape_placeholder = parse_shape(input_symbol, 'a b c d')
        shape = {}
        for name, symbol in shape_placeholder.items():
            shape[name] = symbol if isinstance(symbol, int) \
                else self.backend.eval_symbol(symbol, [(input_symbol, numpy.zeros([10, 20, 30, 40]))])
        print(shape)
        result_placeholder = rearrange(input_symbol, 'a b (c1 c2) (d1 d2) -> (a b d1) c1 (c2 d2)',
                                       **parse_shape(input_symbol, 'a b c1 _'), d2=2)
        result = self.backend.eval_symbol(result_placeholder, [(input_symbol, numpy.zeros([10, 20, 30, 40]))])
        print(result.shape)
        assert result.shape == (10 * 20 * 20, 30, 1 * 2)
        assert numpy.allclose(result, 0)

    @parameterized.expand([
        ([10, 20], [None, None], '...', dict()),
        ([10], [None], '... a', dict(a=10)),
        ([10, 20], [None], '... a', dict(a=20)),
        ([10, 20, 30], [None, None, None], '... a', dict(a=30)),
        ([10, 20, 30, 40], [None, None, None, None], '... a', dict(a=40)),
        ([10], [None], 'a ...', dict(a=10)),
        ([10, 20], [None, None], 'a ...', dict(a=10)),
        ([10, 20, 30], [None, None, None], 'a ...', dict(a=10)),
        ([10, 20, 30, 40], [None, None, None, None], 'a ...', dict(a=10)),
        ([10, 20, 30, 40], [None, None, None, None], ' a ... b', dict(a=10, b=40)),
        ([10, 40], [None, None], ' a ... b', dict(a=10, b=40)),
    ])
    def test_ellipsis(self, static_shape: List[int], shape: List[Optional[int]],
                      pattern: str, expected: Dict[str, int]):
        if self.backend.framework_name in ['mxnet.symbol']:
            # mxnet can't normally run inference
            shape = static_shape
        input_symbol = self.backend.create_symbol(shape)
        shape_placeholder = parse_shape(input_symbol, pattern)
        out_shape = {}
        for name, symbol in shape_placeholder.items():
            if isinstance(symbol, int):
                out_shape[name] = symbol
            else:
                out_shape[name] = self.backend.eval_symbol(
                    symbol, [(input_symbol, numpy.zeros(static_shape))])
        assert out_shape == expected


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
            if 'mindspore' in backend.framework_name and not is_float:
                continue  # mindspore doesn't allow non-floating tensors
            assert backend.is_float_type(input) == is_float, (dtype, backend, input.dtype)
