from einops import transpose, reduce, parse_shape
import numpy
import torch
import mxnet
import cupy
import chainer
import tensorflow as tf

use_tf_eager = False
if use_tf_eager:
    tf.enable_eager_execution()


def mxnet_from_numpy(x):
    return mxnet.ndarray.array([x] if x.shape == () else x)


def chainer_from_numpy(x):
    return chainer.Variable(cupy.asarray(x, dtype='float64'))


def tf_wrap_and_compute(function):
    def returned(x, *args, **kargs):
        x_placeholder = tf.placeholder(dtype=x.dtype)
        return tf.Session().run([function(x_placeholder, *args, **kargs)], {x_placeholder: x})[0]
    return returned


# from, to, transposition, reduction
numpy_functions = (lambda x: x, lambda x: x, transpose, reduce)
torch_functions = (lambda x: torch.from_numpy(x), lambda x: x.numpy(), transpose, reduce)
cupy_functions = (cupy.asarray, cupy.asnumpy, transpose, reduce)
chainer_functions = (chainer_from_numpy, lambda x: cupy.asnumpy(x.data), transpose, reduce)
mxnet_functions = (mxnet_from_numpy, lambda x: x.asnumpy(), transpose, reduce)
tf_eager_functions = (lambda x: tf.contrib.eager.Variable(x), lambda x: x.numpy(), transpose, reduce)
tf_static_functions = (lambda x: x, lambda x: x, tf_wrap_and_compute(transpose), tf_wrap_and_compute(reduce))

framework_functions = dict(
    numpy=numpy_functions,
    cupy=cupy_functions,
    mxnet=mxnet_functions,
    torch=torch_functions,
    chainer=chainer_functions,
)

if use_tf_eager:
    framework_functions['tf_eager'] = tf_eager_functions
else:
    framework_functions['tf_static'] = tf_static_functions


def transpose_numpy_tests():
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


transpose_numpy_tests()


def reduction_tests(functions):
    from_numpy, to_numpy, _transpose, reduce_tensor = functions
    for reduction in ['min', 'max', 'sum']:
        for n_axes in range(7):
            shape = numpy.random.randint(2, 4, size=n_axes)
            permutation = numpy.random.permutation(n_axes)
            skipped = numpy.random.randint(n_axes + 1)
            left = ' '.join(f'x{i}' for i in range(n_axes))
            right = ' '.join(f'x{i}' for i in permutation[skipped:])
            x = numpy.arange(numpy.prod(shape)).reshape(shape)
            result1 = reduce(x, left + '->' + right, operation=reduction)
            result2 = getattr(x.transpose(permutation), reduction)(axis=tuple(range(skipped)))
            result3 = to_numpy(reduce_tensor(from_numpy(x), left + '->' + right, operation=reduction))
            assert numpy.allclose(result1, result2)
            assert numpy.allclose(result1, result3)

        # composite axes
        x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
        result1 = reduce(x, 'a b c d e -> (e c) a', operation=reduction)
        result2 = getattr(x, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1, 2)
        assert numpy.allclose(result1, result2)

        x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
        result1 = reduce(x, 'a b c d e -> (e c a)', operation=reduction)
        result2 = getattr(x, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape(-1)
        assert numpy.allclose(result1, result2)

        x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
        result1 = reduce(x, 'a b c d e -> ', operation=reduction)
        result2 = getattr(x, reduction)()
        assert numpy.allclose(result1, result2)

        # TODO checks for mean, prod and logaddexp (maybe also needed for any and all)


for name, functions in framework_functions.items():
    print('Reduction tests for ', name)
    reduction_tests(functions)


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

    for framework_name, (from_numpy, to_numpy, _transpose, _) in framework_functions.items():
        print('testing examples for ', framework_name)
        for test in tests:
            x = numpy.arange(10 * 20 * 30 * 40).reshape([10, 20, 30, 40])
            result1 = test(x)
            result2 = to_numpy(test(from_numpy(x)))
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
