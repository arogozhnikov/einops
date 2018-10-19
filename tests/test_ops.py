import itertools

import numpy

from einops.einops import (rearrange, reduce, parse_shape, _enumerate_directions, _reductions)
from . import collect_test_backends

imp_op_backends = collect_test_backends(symbolic=False, layers=False)
sym_op_backends = collect_test_backends(symbolic=True, layers=False)

identity_patterns = [
    '...->...',
    'a b c d e-> a b c d e',
    'a b c d e ...-> ... a b c d e',
    'a b c d e ...-> a ... b c d e',
    '... a b c d e -> ... a b c d e',
    'a ... e-> a ... e',
    'a ... -> a ... ',
]

equivalent_rearrange_patterns = [
    ('a b c d e -> (a b) c d e', 'a b ... -> (a b) ... '),
    ('a b c d e -> a b (c d) e', '... c d e -> ... (c d) e'),
    ('a b c d e -> a b c d e', '... -> ... '),
]

equivalent_reduction_patterns = [
    ('a b c d e -> ', ' ... ->  '),
    ('a b c d e -> (e a)', 'a ... e -> (e a)'),
    ('a b c d e -> d (a e)', ' a b c d e ... -> d (a e) '),
]


def test_ellipsis_ops_numpy():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in identity_patterns:
        assert numpy.array_equal(x, rearrange(x, pattern)), pattern

    for pattern1, pattern2 in equivalent_rearrange_patterns:
        assert numpy.array_equal(rearrange(x, pattern1), rearrange(x, pattern2))

    for reduction in ['min', 'max', 'sum']:
        for pattern1, pattern2 in equivalent_reduction_patterns:
            assert numpy.array_equal(reduce(x, pattern1, reduction=reduction),
                                     reduce(x, pattern2, reduction=reduction))

    # now just check coincidence with numpy
    all_rearrange_patterns = [*identity_patterns]
    for pattern_pairs in equivalent_rearrange_patterns:
        all_rearrange_patterns.extend(pattern_pairs)


def check_op_against_numpy(backend, numpy_input, pattern, axes_lengths, reduction='none', is_symbolic=False):
    """
    Helper to test result of operation (rearrange or transpose) against numpy
    if reduction == 'none', rearrange is tested, otherwise
    """

    def operation(x):
        if reduction == 'none':
            return rearrange(x, pattern, **axes_lengths)
        else:
            return reduce(x, pattern, reduction, **axes_lengths)

    numpy_result = operation(numpy_input)
    check_equal = numpy.array_equal
    p_none_dimension = 0.5
    if 'mxnet' in backend.framework_name:
        # known mxnet bug cant work with scalars - allclose
        check_equal = numpy.allclose
        # mxnet can't work unless shape is completely specified
        p_none_dimension = 0
    if is_symbolic:
        symbol_shape = [d if numpy.random.random() >= p_none_dimension else None for d in numpy_input.shape]
        symbol = backend.create_symbol(shape=symbol_shape)
        result_symbol = operation(symbol)
        backend_result = backend.eval_symbol(result_symbol, [(symbol, numpy_input)])
    else:
        backend_result = operation(backend.from_numpy(numpy_input))
        backend_result = backend.to_numpy(backend_result)

    check_equal(numpy_result, backend_result)


def test_ellipsis_ops_imperative():
    """ Checking various patterns against numpy """
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for is_symbolic in [True, False]:
        for backend in collect_test_backends(symbolic=is_symbolic, layers=False):
            for pattern in identity_patterns + list(itertools.chain(*equivalent_rearrange_patterns)):
                check_op_against_numpy(backend, x, pattern, axes_lengths={}, is_symbolic=is_symbolic)

            for reduction in ['min', 'max', 'sum']:
                for pattern in itertools.chain(*equivalent_reduction_patterns):
                    check_op_against_numpy(backend, x, pattern,
                                           axes_lengths={}, reduction=reduction, is_symbolic=is_symbolic)


def test_rearrange_consistency_numpy():
    shape = [1, 2, 3, 5, 7, 11]
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    for pattern in [
        'a b c d e f-> a b c d e f',
        'b a c d e f-> a b d e f c',
        'a b c d e f-> f e d c b a',
        'a b c d e f-> (f e) d (c b a)',
        'a b c d e f-> (f e d c b a)',
    ]:
        result = rearrange(x, pattern)
        assert len(numpy.setdiff1d(x, result)) == 0
        assert result.dtype == x.dtype

    result = rearrange(x, 'a b c d e f -> a (b) (c d e) f')
    assert numpy.array_equal(x.flatten(), result.flatten())

    result = rearrange(x, 'a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11')
    assert numpy.array_equal(x, result)

    result1 = rearrange(x, 'a b c d e f -> f e d c b a')
    result2 = rearrange(x, 'f e d c b a -> a b c d e f')
    assert numpy.array_equal(result1, result2)

    result = rearrange(rearrange(x, 'a b c d e f -> (f d) c (e b) a'), '(f d) c (e b) a -> a b c d e f', b=2, d=5)
    assert numpy.array_equal(x, result)

    sizes = dict(zip('abcdef', shape))
    temp = rearrange(x, 'a b c d e f -> (f d) c (e b) a', **sizes)
    result = rearrange(temp, '(f d) c (e b) a -> a b c d e f', **sizes)
    assert numpy.array_equal(x, result)

    x2 = numpy.arange(2 * 3 * 4).reshape([2, 3, 4])
    result = rearrange(x2, 'a b c -> b c a')
    assert x2[1, 2, 3] == result[2, 3, 1]
    assert x2[0, 1, 2] == result[1, 2, 0]


def test_rearrange_numpy_element_wise():
    for n_axes in range(1, 10):
        input = numpy.arange(2 ** n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = ' '.join(f'i{axis}' for axis in range(n_axes))
        right_expression = ' '.join(f'i{axis}' for axis in permutation)
        expression = left_expression + ' -> ' + right_expression
        result = rearrange(input, expression)

        for pick in numpy.random.randint(0, 2, [10, n_axes]):
            assert input[tuple(pick)] == result[tuple(pick[permutation])]

    for n_axes in range(1, 10):
        input = numpy.arange(2 ** n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = ' '.join(f'i{axis}' for axis in range(n_axes)[::-1])
        right_expression = ' '.join(f'i{axis}' for axis in permutation[::-1])
        expression = left_expression + ' -> ' + right_expression
        result = rearrange(input, expression)
        assert result.shape == input.shape
        expected_result = numpy.zeros_like(input)
        for original_axis, result_axis in enumerate(permutation):
            expected_result |= ((input >> original_axis) & 1) << result_axis

        assert numpy.array_equal(result, expected_result)


def test_reduction_imperatives():
    for backend in imp_op_backends:
        print('Reduction tests for ', backend.framework_name)
        for reduction in _reductions:
            input = numpy.arange(2 * 3 * 4 * 5 * 6, dtype='int64').reshape(2, 3, 4, 5, 6)
            if reduction in ['mean', 'prod']:
                input = input / input.astype('float64').mean()
            test_cases = [
                ['a b c d e -> ', {}, getattr(input, reduction)()],
                ['... -> ', {}, getattr(input, reduction)()],
                ['(a1 a2) ... (e1 e2) -> ', dict(a1=1, e2=2), getattr(input, reduction)()],
                ['a b c d e -> (e c) a', {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a ... c d e -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a b c d e ... -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a b c d e -> (e c a)', {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1)],
                ['(a1 a2) ... -> (a2 a1) ...', dict(a2=1), input],
            ]
            for pattern, axes_lengths, expected_result in test_cases:
                result = reduce(backend.from_numpy(input.copy()), pattern, reduction=reduction, **axes_lengths)
                result = backend.to_numpy(result)
                assert numpy.allclose(result, expected_result)


def test_reduction_symbolic():
    for backend in sym_op_backends:
        print('Reduction tests for ', backend.framework_name)
        for reduction in _reductions:
            input = numpy.arange(2 * 3 * 4 * 5 * 6, dtype='int64').reshape(2, 3, 4, 5, 6)
            input = input / input.astype('float64').mean()
            test_cases = [
                ['a b c d e -> ', {},
                 getattr(input, reduction)()],
                ['a ... -> ', {},
                 getattr(input, reduction)()],
                ['(a a2) ... (e e2) -> ', dict(a2=1, e2=1),
                 getattr(input, reduction)()],
                ['a b c d e -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a ... c d e -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a b c d e ... -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a b c d e -> (e c a)', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1)],
                ['(a a2) ... -> (a2 a) ...', dict(a2=1),
                 input],
            ]
            for pattern, axes_lengths, expected_result in test_cases:
                shapes = [input.shape]
                if backend.framework_name != 'mxnet.symbol':
                    # mxnet can't handle non-specified shapes
                    shapes.append([None for _ in input.shape])
                for shape in shapes:
                    sym = backend.create_symbol(shape)
                    result_sym = reduce(sym, pattern, reduction=reduction, **axes_lengths)
                    result = backend.eval_symbol(result_sym, [(sym, input)])
                    assert numpy.allclose(result, expected_result)

                if True:
                    shape = []
                    _axes_lengths = {**axes_lengths}
                    for axis, length in zip('abcde', input.shape):
                        # filling as much as possible with Nones
                        if axis in pattern:
                            shape.append(None)
                            _axes_lengths[axis] = length
                        else:
                            shape.append(length)
                    sym = backend.create_symbol(shape)
                    result_sym = reduce(sym, pattern, reduction=reduction, **_axes_lengths)
                    result = backend.eval_symbol(result_sym, [(sym, input)])
                    assert numpy.allclose(result, expected_result)


def test_reduction_stress_imperatives():
    for backend in imp_op_backends:
        print('Stress-testing reduction for ', backend.framework_name)
        for reduction in _reductions + ('none',):
            dtype = 'int64'
            coincide = numpy.array_equal
            if reduction in ['mean', 'prod']:
                dtype = 'float64'
                coincide = numpy.allclose
            for n_axes in range(7):
                shape = numpy.random.randint(2, 4, size=n_axes)
                permutation = numpy.random.permutation(n_axes)
                skipped = 0 if reduction == 'none' else numpy.random.randint(n_axes + 1)
                left = ' '.join(f'x{i}' for i in range(n_axes))
                right = ' '.join(f'x{i}' for i in permutation[skipped:])
                pattern = left + '->' + right
                x = numpy.arange(1, 1 + numpy.prod(shape), dtype=dtype).reshape(shape)
                if reduction == 'prod':
                    x /= x.mean()
                result1 = reduce(x, pattern, reduction=reduction)
                result2 = x.transpose(permutation)
                if skipped > 0:
                    result2 = getattr(result2, reduction)(axis=tuple(range(skipped)))
                assert coincide(result1, result2)
                if n_axes == 0 and 'mxnet' in backend.framework_name:
                    # known mxnet bug, cant attach gradients to scalar
                    continue
                check_op_against_numpy(backend, x, pattern, reduction=reduction, axes_lengths={}, is_symbolic=False)


def test_rearrange_examples():
    # TODO order
    # transposition = permute_dimensions
    # reshape = view
    # squeeze, unsqueeze
    # concatenating and stacking
    # depth-to-space and space-to-depth
    # splitting of dimension into groups
    # stack and concat

    # ну и всевозможные редукции

    # shufflenet reordering
    # max-pooling
    # strided convolutions (1d 2d)
    # добавление / вытаскивание глубины для одномерных моделей
    # отрисовка набора изображений

    def test1(x):
        y = rearrange(x, 'b h w c -> b c h w')
        assert y.shape == (10, 40, 20, 30)
        return y

    def test2(x):
        y = rearrange(x, 'b h w c -> b c (h w)')
        assert y.shape == (10, 40, 20 * 30)
        return y

    def test3(x):
        y = rearrange(x, 'b h w (c h1 w1) -> b (h h1) (w w1) c', h1=2, w1=2)
        assert y.shape == (10, 40, 60, 10)
        return y

    def test4(x):
        y = rearrange(x, 'b (h h1) (w w1) c -> b h w (h1 w1 c)', h1=2, w1=2)
        assert y.shape == (10, 10, 15, 160)
        return y

    def test5(x):
        y = rearrange(x, 'b1 s b2 t -> b1 b2 s t')
        assert y.shape == (10, 30, 20, 40)
        return y

    def test6(x):
        # TODO return matrix-by-matrix multiplication
        t = rearrange(x, 'b c h w -> (b h w) c')
        assert t.shape == (10 * 30 * 40, 20)

        # TODO this test specifically for TF with x.shape replaced by tf.shape for expression
        y = rearrange(t, '(b h w) c2 -> b c2 h w', **parse_shape(x, 'b _ h w'))
        assert y.shape == (10, 20, 30, 40)
        return y

    def test7(x):
        y1, y2 = rearrange(x, 'b h w (c g) -> g b h w c', g=2)
        assert y1.shape == (10, 20, 30, 20)
        assert y2.shape == (10, 20, 30, 20)
        return y1 + y2

    tests = [test1, test2, test3, test4, test5, test6]

    for backend in imp_op_backends:
        print('testing examples for ', backend.framework_name)
        if 'tensorflow' in backend.framework_name:
            extended_tests = tests
        else:
            extended_tests = tests + [test7]
        for test in extended_tests:
            x = numpy.arange(10 * 20 * 30 * 40).reshape([10, 20, 30, 40])
            result1 = test(x)
            result2 = backend.to_numpy(test(backend.from_numpy(x)))
            assert numpy.array_equal(result1, result2)

            # now with strides
            x = numpy.arange(10 * 2 * 20 * 3 * 30 * 1 * 40).reshape([10 * 2, 20 * 3, 30 * 1, 40])
            # known torch bug - torch doesn't support negative steps
            last_step = -1 if backend.framework_name != 'torch' else 1
            indexing_expression = numpy.index_exp[::2, ::3, ::1, ::last_step]
            result1 = test(x[indexing_expression])
            result2 = backend.to_numpy(test(backend.from_numpy(x)[indexing_expression]))
            assert numpy.array_equal(result1, result2)

    def shufflenet(x, convolve, c1=8, c2=8):
        # shufflenet example
        x = convolve(x)
        x = rearrange(x, 'b (c1 c2) h w-> b (c2 c1) h w', c1=c1, c2=c2)
        x = convolve(x)
        print(x.shape)

    def convolve_strided_1d(x, stride, usual_conv):
        x_reshaped = rearrange(x, 'b c (t stride) -> (stride b) c t)', stride=stride)
        result = usual_conv(x_reshaped)
        return rearrange(result, '(stride b) c t -> b c (t stride)')

    def convolve_strided_2d(x, stride, usual_conv):
        x_reshaped = rearrange(x, 'b c (h h1) (w w1) -> (h1 w1 b) c h w)', stride=stride)
        result = usual_conv(x_reshaped)
        return rearrange(result, '(h1 w1 b) c h w) -> b c (h h1) (w w1)')

    # TODO example for detection module?

    # TODO example for tensor train?
    # einsum(  G[i, j, alpha0, alpha1] X[...,  i, alpha0] -> [i, ...,  alpha1]  )


def test_enumerating_directions():
    for backend in imp_op_backends:
        print('testing directions for', backend.framework_name)
        for shape in [[], [1], [1, 1, 1], [2, 3, 5, 7]]:
            if backend.framework_name == 'mxnet.ndarray' and len(shape) == 0:
                # known bug of mxnet
                continue
            x = numpy.arange(numpy.prod(shape)).reshape(shape)
            axes1 = _enumerate_directions(x)
            axes2 = _enumerate_directions(backend.from_numpy(x))
            for axe1, axe2 in zip(axes1, axes2):
                axe2 = backend.to_numpy(axe2)
                assert axe1.shape == axe2.shape
                assert numpy.allclose(axe1, axe2)


def test_concatenations_and_stacking():
    for backend in imp_op_backends:
        print('testing shapes for ', backend.framework_name)
        for n_arrays in [1, 2, 5]:
            shapes = [[], [1], [1, 1], [2, 3, 5, 7], [1] * 6]
            for shape in shapes:
                if backend.framework_name == 'mxnet.ndarray' and len(shape) == 0:
                    # known bug of mxnet
                    continue
                arrays1 = [numpy.arange(i, i + numpy.prod(shape)).reshape(shape) for i in range(n_arrays)]
                arrays2 = [backend.from_numpy(array) for array in arrays1]
                result0 = numpy.asarray(arrays1)
                result1 = rearrange(arrays1, '...->...')
                result2 = rearrange(arrays2, '...->...')
                assert numpy.array_equal(result0, result1)
                assert numpy.array_equal(result1, backend.to_numpy(result2))

                result1 = rearrange(arrays1, 'b ... -> ... b')
                result2 = rearrange(arrays2, 'b ... -> ... b')
                assert numpy.array_equal(result1, backend.to_numpy(result2))


def test_gradients_imperatives():
    # lazy - just checking reductions
    for reduction in _reductions:
        x = numpy.arange(1, 1 + 2 * 3 * 4).reshape(2, 3, 4).astype('float32')
        results = {}
        for backend in imp_op_backends:
            y0 = backend.from_numpy(x)
            if not hasattr(y0, 'grad'):
                continue
            if 'mxnet' in backend.framework_name:
                backend.mx.autograd.set_recording(True)
            y1 = reduce(y0, 'a b c -> c a', reduction=reduction)
            y2 = reduce(y1, 'c a -> a c', reduction=reduction)
            y3 = reduce(y2, 'a (c1 c2) -> a', reduction=reduction, c1=2)
            y4 = reduce(y3, '... -> ', reduction=reduction)
            if 'mxnet' in backend.framework_name:
                backend.mx.autograd.set_recording(False)
            y4.backward()
            grad = backend.to_numpy(y0.grad)
            results[backend.framework_name] = grad

        print('comparing gradients for', results.keys())
        for name1, grad1 in results.items():
            for name2, grad2 in results.items():
                assert numpy.allclose(grad1, grad2), [name1, name2, 'provided different gradients']

# will not work that easily with nosetests
# print(_prepare_transformation_recipe.cache_info())
