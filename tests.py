from einops import transpose, reduce, parse_shape, enumerate_directions
import numpy
import tensorflow as tf
import backends

use_tf_eager = True
if use_tf_eager:
    tf.enable_eager_execution()

all_backends = [c() for c in backends.AbstractBackend.__subclasses__()]


def test_transpose_ellipsis_numpy():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    assert (numpy.allclose(x, transpose(x, '...->...')))
    assert (numpy.allclose(x, transpose(x, 'a b c d e-> a b c d e')))
    assert (numpy.allclose(x, transpose(x, 'a b c d e ...-> ... a b c d e')))
    assert (numpy.allclose(x, transpose(x, 'a b c d e ...-> a ... b c d e')))
    assert (numpy.allclose(x, transpose(x, '... a b c d e -> ... a b c d e')))
    assert (numpy.allclose(x, transpose(x, 'a ... e-> a ... e')))
    assert (numpy.allclose(x, transpose(x, 'a ... -> a ... ')))
    assert (numpy.allclose(x, transpose(x, 'a ... -> a ... ')))

    assert (numpy.allclose(transpose(x, 'a b c d e -> (a b) c d e'),
                           transpose(x, 'a b ... -> (a b ) ... ')))
    assert (numpy.allclose(transpose(x, 'a b c d e -> a b (c d) e'),
                           transpose(x, '... c d e -> ... (c d) e')))
    assert (numpy.allclose(transpose(x, 'a b c d e -> a b c d e'),
                           transpose(x, '... -> ... ')))
    for reduction in ['min', 'max', 'sum']:
        assert (numpy.allclose(reduce(x, 'a b c d e -> ', operation=reduction),
                               reduce(x, '... -> ', operation=reduction)))
        assert (numpy.allclose(reduce(x, 'a b c d e -> (e a)', operation=reduction),
                               reduce(x, 'a ... e -> (e a)', operation=reduction)))
        assert (numpy.allclose(reduce(x, 'a b c d e -> d (a e)', operation=reduction),
                               reduce(x, 'a b c d e ... -> d (a e)', operation=reduction)))
    # TODO ellipsis inside parentheses on the right side
    # assert (numpy.allclose(transpose(x, 'a b c d e -> (a b c d e)'),
    #                        transpose(x, '... -> (...) ')))


test_transpose_ellipsis_numpy()


def test_transpose_with_numpy():
    shape = [1, 1, 2, 3, 5, 8]
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    for expression in [
        'a b c d e f-> a b c d e f',
        'b a c d e f-> a b d e f c',
        'a b c d e f-> f e d c b a',
        'a b c d e f-> (f e) d (c b a)',
        'a b c d e f-> (f e d c b a)',
    ]:
        result = transpose(x, expression)
        assert len(numpy.setdiff1d(x, result)) == 0
        assert result.dtype == x.dtype

    result = transpose(x, 'a b c d e f -> a b c d e f')
    assert numpy.allclose(x, result)

    result = transpose(x, 'a b c d e f -> a (b) (c d e) f')
    assert numpy.allclose(x.flatten(), result.flatten())

    result = transpose(x, 'a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11')
    assert numpy.allclose(x, result)

    result1 = transpose(x, 'a b c d e f -> f e d c b a')
    result2 = transpose(x, 'f e d c b a -> a b c d e f')
    assert numpy.allclose(result1, result2)

    result = transpose(transpose(x, 'a b c d e f -> (f d) c (e b) a'), '(f d) c (e b) a -> a b c d e f', b=1, d=3)
    assert numpy.allclose(x, result)

    sizes = dict(zip('abcdef', shape))
    temp = transpose(x, 'a b c d e f -> (f d) c (e b) a', **sizes)
    result = transpose(temp, '(f d) c (e b) a -> a b c d e f', **sizes)
    assert numpy.allclose(x, result)

    x2 = numpy.arange(2 * 3 * 4).reshape([2, 3, 4])
    result = transpose(x2, 'a b c -> b c a')
    assert x2[1, 2, 3] == result[2, 3, 1]
    assert x2[0, 1, 2] == result[1, 2, 0]

    for n_axes in range(1, 10):
        input = numpy.arange(2 ** n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = ' '.join(f'i{axis}' for axis in range(n_axes))
        right_expression = ' '.join(f'i{axis}' for axis in permutation)
        expression = left_expression + ' -> ' + right_expression
        result = transpose(input, expression)

        for pick in numpy.random.randint(0, 2, [10, n_axes]):
            assert input[tuple(pick)] == result[tuple(pick[permutation])]

    for n_axes in range(1, 10):
        input = numpy.arange(2 ** n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = ' '.join(f'i{axis}' for axis in range(n_axes)[::-1])
        right_expression = ' '.join(f'i{axis}' for axis in permutation[::-1])
        expression = left_expression + ' -> ' + right_expression
        result = transpose(input, expression)
        assert result.shape == input.shape
        expected_result = numpy.zeros_like(input)
        for original_axis, result_axis in enumerate(permutation):
            expected_result |= ((input >> original_axis) & 1) << result_axis

        assert numpy.allclose(result, expected_result)
    print('simple tests passed')


test_transpose_with_numpy()


def test_reduction():
    for backend in all_backends:
        print('Reduction tests for ', backend.framework_name)
        # TODO checks for logaddexp (maybe also needed for any and all)
        for reduction in ['min', 'max', 'sum', 'mean', 'prod']:
            dtype = 'float64' if reduction == 'mean' else 'int64'
            # composite axes
            x = numpy.arange(2 * 3 * 4 * 5 * 6, dtype=dtype).reshape(2, 3, 4, 5, 6)
            result1 = reduce(x, 'a b c d e -> (e c) a', operation=reduction)
            result2 = getattr(x, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)
            assert numpy.allclose(result1, result2)

            x = numpy.arange(2 * 3 * 4 * 5 * 6, dtype=dtype).reshape(2, 3, 4, 5, 6)
            result1 = reduce(x, 'a b c d e -> (e c a)', operation=reduction)
            result2 = getattr(x, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1)
            assert numpy.allclose(result1, result2)

            x = numpy.arange(2 * 3 * 4 * 5 * 6, dtype=dtype).reshape(2, 3, 4, 5, 6)
            result1 = reduce(x, 'a b c d e -> ', operation=reduction)
            result2 = getattr(x, reduction)()
            assert numpy.allclose(result1, result2)

            x = numpy.arange(2 * 3 * 4 * 5 * 6, dtype=dtype).reshape(2, 3, 4, 5, 6)
            result1 = reduce(x, '... -> ', operation=reduction)
            result2 = getattr(x, reduction)()
            assert numpy.allclose(result1, result2)

            x = numpy.arange(2 * 3 * 4 * 5 * 6, dtype=dtype).reshape(2, 3, 4, 5, 6)
            result1 = reduce(x, '(a1 a2) ... (e1 e2) -> ', operation=reduction, a1=2, e1=2)
            result2 = getattr(x, reduction)()
            assert numpy.allclose(result1, result2)


test_reduction()


def test_reduction_stress():
    for backend in all_backends:
        print('Stress-testing reduction for ', backend.framework_name)
        if 'mxnet' in backend.framework_name:
            print('skipped testing for mxnet, not working yet with scalar tensors')
            continue
        print('Reduction tests for ', backend.framework_name)
        for reduction in ['min', 'max', 'sum', 'mean', 'prod']:
            dtype = 'int64'
            if reduction in ['mean', 'prod']:
                dtype = 'float64'
            for n_axes in range(7):
                shape = numpy.random.randint(2, 4, size=n_axes)
                permutation = numpy.random.permutation(n_axes)
                skipped = numpy.random.randint(n_axes + 1)
                left = ' '.join(f'x{i}' for i in range(n_axes))
                right = ' '.join(f'x{i}' for i in permutation[skipped:])
                x = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
                if reduction == 'prod' and x.shape != ():
                    x /= x.mean()
                result1 = reduce(x, left + '->' + right, operation=reduction)
                result2 = getattr(x.transpose(permutation), reduction)(axis=tuple(range(skipped)))
                result3 = backend.to_numpy(reduce(backend.from_numpy(x), left + '->' + right, operation=reduction))
                assert numpy.allclose(result1, result2)
                assert numpy.allclose(result1, result3)


test_reduction_stress()


def test_transpose_examples():
    def test1(x):
        y = transpose(x, 'b h w c -> b c h w')
        assert y.shape == (10, 40, 20, 30)
        return y

    def test2(x):
        y = transpose(x, 'b h w c -> b c (h w)')
        assert y.shape == (10, 40, 20 * 30)
        return y

    def test3(x):
        y = transpose(x, 'b h w (c h1 w1) -> b (h h1) (w w1) c', h1=2, w1=2)
        assert y.shape == (10, 40, 60, 10)
        return y

    def test4(x):
        y = transpose(x, 'b (h h1) (w w1) c -> b h w (h1 w1 c)', h1=2, w1=2)
        assert y.shape == (10, 10, 15, 160)
        return y

    def test5(x):
        y1, y2 = transpose(x, 'b h w (c g) -> g b h w c', g=2)
        assert y1.shape == (10, 20, 30, 20)
        assert y2.shape == (10, 20, 30, 20)
        return y1 + y2

    def test6(x):
        y = transpose(x, 'b1 s b2 t -> b1 b2 s t')
        assert y.shape == (10, 30, 20, 40)
        return y

    def test7(x):
        # TODO return matrix-by-matrix multiplication
        t = transpose(x, 'b c h w -> (b h w) c')
        assert t.shape == (10 * 30 * 40, 20)

        # TODO this test specifically for TF with x.shape replaced by tf.shape for expression
        y = transpose(t, '(b h w) c2->b c2 h w', **parse_shape(x, 'b _ h w'))
        assert y.shape == (10, 20, 30, 40)
        return y

    tests = [test1, test2, test3, test4, test5, test6, test7]

    for backend in all_backends:
        print('testing examples for ', backend.framework_name)
        for test in tests:
            x = numpy.arange(10 * 20 * 30 * 40).reshape([10, 20, 30, 40])
            result1 = test(x)
            result2 = backend.to_numpy(test(backend.from_numpy(x)))
            assert numpy.allclose(result1, result2)
            print(result1.shape)

    def shufflenet(x, convolve, c1=8, c2=8):
        # shufflenet example
        x = convolve(x)
        x = transpose(x, 'b (c1 c2) h w-> b (c2 c1) h w', c1=c1, c2=c2)
        x = convolve(x)
        print(x.shape)

    def convolve_strided_1d(x, stride, usual_conv):
        x_reshaped = transpose(x, 'b c (t stride) -> (stride b) c t)', stride=stride)
        result = usual_conv(x_reshaped)
        return transpose(result, '(stride b) c t -> b c (t stride)')

    def convolve_strided_2d(x, stride, usual_conv):
        x_reshaped = transpose(x, 'b c (h h1) (w w1) -> (h1 w1 b) c h w)', stride=stride)
        result = usual_conv(x_reshaped)
        return transpose(result, '(h1 w1 b) c h w) -> b c (h h1) (w w1)')

    # TODO add example for detection module

    # tensor train
    # einsum(  G[i, j, alpha0, alpha1] X[...,  i, alpha0] -> [i, ...,  alpha1]  )


test_transpose_examples()


def test_parse_shape():
    for backend in all_backends:
        # TODO skip static tensorflow
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


def test_parse_shape_tf_static():
    print('special shape parsing for tf_static')
    placeholders = [
        tf.placeholder('float32', [10, 20, 30, 40]),
        tf.placeholder('float32', [10, 20, None, None]),
        tf.placeholder('float32', [None, None, None, None]),
    ]
    for placeholder in placeholders:
        shape_placeholder = parse_shape(placeholder, 'a b c d')
        shape = tf.Session().run(shape_placeholder, {placeholder: numpy.zeros([10, 20, 30, 40])})
        print(shape)

        result_placeholder = transpose(placeholder, 'a b (c1 c2) (d1 d2) -> (a b d1) c1 (c2 d2)',
                                       **parse_shape(placeholder, 'a b c1 _'), d2=2)
        result = tf.Session().run(result_placeholder, {placeholder: numpy.zeros([10, 20, 30, 40])})
        print(result.shape)
        assert result.shape == (10 * 20 * 20, 30, 1 * 2)
        assert numpy.allclose(result, 0)


test_parse_shape()
if not use_tf_eager:
    test_parse_shape_tf_static()


def test_enumerating_directions():
    for backend in all_backends:
        print('testing directions for', backend.framework_name)
        for shape in [[], [1], [1, 1, 1], [2, 3, 5, 7]]:
            if backend.framework_name == 'mxnet.ndarray' and len(shape) == 0:
                # known bug of ndarray
                continue
            x = numpy.arange(numpy.prod(shape)).reshape(shape)
            axes1 = enumerate_directions(x)
            axes2 = enumerate_directions(backend.from_numpy(x))
            for axe1, axe2 in zip(axes1, axes2):
                axe2 = backend.to_numpy(axe2)
                assert axe1.shape == axe2.shape
                assert numpy.allclose(axe1, axe2)


test_enumerating_directions()


def test_concatenations():
    for backend in all_backends:
        print('testing shapes for ', backend.framework_name)
        for n_arrays in [1, 2, 5]:
            for shape in [[], [1], [1, 1], [2, 3, 5, 7], [1] * 6]:
                if shape == [] and backend.framework_name == 'mxnet.ndarray':
                    continue
                arrays1 = [numpy.arange(i, i + numpy.prod(shape)).reshape(shape) for i in range(n_arrays)]
                arrays2 = [backend.from_numpy(array) for array in arrays1]
                result1 = transpose(arrays1, '...->...')
                result2 = transpose(arrays2, '...->...')
                assert numpy.allclose(result1, backend.to_numpy(result2))


test_concatenations()
