import numpy

from einops.einops import (rearrange, reduce, parse_shape,
                           _enumerate_directions, _optimize_transformation, _reductions,
                           _check_elementary_axis_name)
from . import collect_test_backends

imp_op_backends = collect_test_backends(symbolic=False, layers=False)
sym_op_backends = collect_test_backends(symbolic=True, layers=False)


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


def test_elementary_axis_name():
    for name in ['a', 'b', 'h', 'dx', 'h1', 'zz', 'i9123', 'somelongname']:
        assert _check_elementary_axis_name(name)
    for name in ['', '2b', 'Alex', 'camelCase', 'under_score', '12']:
        assert not _check_elementary_axis_name(name)


def test_rearrange_ellipsis_numpy():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    assert numpy.array_equal(x, rearrange(x, '...->...'))
    assert numpy.array_equal(x, rearrange(x, 'a b c d e-> a b c d e'))
    assert numpy.array_equal(x, rearrange(x, 'a b c d e ...-> ... a b c d e'))
    assert numpy.array_equal(x, rearrange(x, 'a b c d e ...-> a ... b c d e'))
    assert numpy.array_equal(x, rearrange(x, '... a b c d e -> ... a b c d e'))
    assert numpy.array_equal(x, rearrange(x, 'a ... e-> a ... e'))
    assert numpy.array_equal(x, rearrange(x, 'a ... -> a ... '))

    assert numpy.array_equal(rearrange(x, 'a b c d e -> (a b) c d e'),
                             rearrange(x, 'a b ... -> (a b) ... '))
    assert numpy.array_equal(rearrange(x, 'a b c d e -> a b (c d) e'),
                             rearrange(x, '... c d e -> ... (c d) e'))
    assert numpy.array_equal(rearrange(x, 'a b c d e -> a b c d e'),
                             rearrange(x, '... -> ... '))
    for reduction in ['min', 'max', 'sum']:
        assert numpy.array_equal(reduce(x, 'a b c d e -> ', reduction=reduction),
                                 reduce(x, '... -> ', reduction=reduction))
        assert numpy.array_equal(reduce(x, 'a b c d e -> (e a)', reduction=reduction),
                                 reduce(x, 'a ... e -> (e a)', reduction=reduction))
        assert numpy.array_equal(reduce(x, 'a b c d e -> d (a e)', reduction=reduction),
                                 reduce(x, 'a b c d e ... -> d (a e)', reduction=reduction))


def test_rearrange_with_numpy():
    shape = [1, 2, 3, 5, 7, 11]
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    for expression in [
        'a b c d e f-> a b c d e f',
        'b a c d e f-> a b d e f c',
        'a b c d e f-> f e d c b a',
        'a b c d e f-> (f e) d (c b a)',
        'a b c d e f-> (f e d c b a)',
    ]:
        result = rearrange(x, expression)
        assert len(numpy.setdiff1d(x, result)) == 0
        assert result.dtype == x.dtype

    result = rearrange(x, 'a b c d e f -> a b c d e f')
    assert numpy.array_equal(x, result)

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
            input /= input.astype('float64').mean()
            test_cases = [
                ['a b c d e -> ', {}, getattr(input, reduction)()],
                ['a ... -> ', {}, getattr(input, reduction)()],
                ['(a a2) ... (e e2) -> ', dict(a2=1, e2=1), getattr(input, reduction)()],
                ['a b c d e -> (e c) a', {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a ... c d e -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a b c d e ... -> (e c) a', {},
                 getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)],
                ['a b c d e -> (e c a)', {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1)],
                ['(a a2) ... -> (a2 a) ...', dict(a2=1), input],
            ]
            for pattern, axes_lengths, expected_result in test_cases:
                shapes = [input.shape]
                if backend.framework_name != 'mxnet.symbol':
                    shapes.append(tuple([None] * 5))
                for shape in shapes:
                    sym = backend.create_symbol(shape)
                    result_sym = reduce(sym, pattern, reduction=reduction, **axes_lengths)
                    result = backend.eval_symbol(result_sym, [(sym, input)])
                    assert numpy.allclose(result, expected_result)

                if True:
                    shape = []
                    _axes_lengths = {**axes_lengths}
                    for axis, length in zip('abcde', input.shape):
                        # filling as much with Nones
                        if axis in pattern:
                            shape.append(None)
                            _axes_lengths[axis] = length
                        else:
                            shape.append(length)
                    sym = backend.create_symbol(shape)
                    result_sym = reduce(sym, pattern, reduction=reduction, **_axes_lengths)
                    result = backend.eval_symbol(result_sym, [(sym, input)])
                    assert numpy.allclose(result, expected_result)


def test_reduction_stress():
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
                if len(right) == 0 and ('mxnet' in backend.framework_name):
                    # known mxnet bug
                    continue
                x = numpy.arange(1, 1 + numpy.prod(shape), dtype=dtype).reshape(shape)
                if reduction == 'prod':
                    x /= x.mean()
                result1 = reduce(x, left + '->' + right, reduction=reduction)
                result2 = x.transpose(permutation)
                if skipped > 0:
                    result2 = getattr(result2, reduction)(axis=tuple(range(skipped)))
                result3 = backend.to_numpy(reduce(backend.from_numpy(x), left + '->' + right, reduction=reduction))
                assert coincide(result1, result2)
                assert coincide(result1, result3)


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
    #

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


def test_parse_shape_imperative():
    for backend in imp_op_backends:
        print('Shape parsing for ', backend.framework_name)
        x = numpy.zeros([10, 20, 30, 40])
        parsed1 = parse_shape(x, 'a b c d')
        parsed2 = parse_shape(backend.from_numpy(x), 'a b c d')
        print(parsed2)
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
    # TODO add sym layer backends?
    for backend in sym_op_backends:
        print('special shape parsing for', backend.framework_name)
        input_symbols = [
            backend.create_symbol([10, 20, 30, 40]),
            backend.create_symbol([10, 20, None, None]),
            backend.create_symbol([None, None, None, None]),
        ]
        if backend.framework_name == 'mxnet.symbol':
            # mxnet can't normally run inference
            input_symbols = [backend.create_symbol([10, 20, 30, 40])]

        for input_symbol in input_symbols:
            print(input_symbol)
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


def test_doctests():
    from doctest import testmod
    import einops
    testmod(einops.einops, raise_on_error=True, )
    import einops.layers
    testmod(einops.layers, raise_on_error=True)

# TODO test for gradients

# will not work that easily with nosetests
# print(_prepare_transformation_recipe.cache_info())
