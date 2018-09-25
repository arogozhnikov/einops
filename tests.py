from einops import transpose as transpose_1, transpose_2
import numpy
import torch
import mxnet
import cupy
import chainer


def transpose(*args, **kargs):
    result1 = transpose_1(*args, **kargs)
    if isinstance(result1, numpy.ndarray):
        result2 = transpose_2(*args, **kargs)
        assert (result1 - result2).max() < 1e-5 * result1.max()
        assert (result2 - result1).max() < 1e-5 * result1.max()
    return result1


def simple_tests():
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
        print('simple tests passed')
        assert result.dtype == x.dtype

    result = transpose(x, 'a b c d e f -> a b c d e f')
    assert numpy.allclose(x, result)

    result = transpose(x, 'a b c d e f -> a (b) (c d e) f')
    assert numpy.allclose(x.flatten(), result.flatten())

    result1 = transpose(x, 'a b c d e f -> f e d c b a')
    result2 = transpose(x, 'f e d c b a -> a b c d e f')
    assert numpy.allclose(result1, result2)

    result = transpose(transpose(x, 'a b c d e f -> (f d) c (e b) a'), '(f d) c (e b) a -> a b c d e f', b=1, d=3)
    assert numpy.allclose(x, result)

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
            # TODO i don't quite get the ordering
            expected_result |= ((input >> original_axis) & 1) << result_axis

        assert numpy.allclose(result, expected_result)


simple_tests()


def test(make_array, transpose):
    x_small = make_array(2, 4)
    x = make_array(10, 20, 30, 40)
    multiplier = make_array(20, 50)

    result1 = transpose(x_small, 'a(bc)->bca', c=2)
    assert result1.shape == (2, 2, 2)

    y = transpose(x, 'b h w c -> b c h w')
    print(y.shape)
    assert y.shape == (10, 40, 20, 30)

    y = transpose(x, 'b h w c -> b c (h w)')
    print(y.shape)
    assert y.shape == (10, 40, 20 * 30)

    y = transpose(x, 'b h w (c h1 w1) -> b (h h1) (w w1) c', h1=2, w1=2)
    print(y.shape)
    assert y.shape == (10, 40, 60, 10)

    y = transpose(x, 'b (h h1) (w w1) c -> b h w (h1 w1 c)', h1=2, w1=2)
    print(y.shape)
    assert y.shape == (10, 10, 15, 160)

    y1, y2 = transpose(x, 'b h w (c g) -> g b h w c', g=2)
    print(y1.shape, y2.shape)
    assert y1.shape == (10, 20, 30, 20)
    assert y2.shape == (10, 20, 30, 20)

    y = transpose(x, 'b1 s b2 t->b1 b2 s t')
    print(y.shape)
    assert y.shape == (10, 30, 20, 40)

    try:
        t = transpose(x, 'b c h w->(b h w) c') @ multiplier  # @ это просто перемножение матриц
        print(t.shape)
        assert t.shape == (10 * 30 * 40, 50)
    except:
        print(type(x), 'does not support dot product ')
        t = make_array(10 * 30 * 40, 50)

    y = transpose(t, '(b h w) c2->b c2 h w', b_hw=x.shape)
    print(y.shape)
    assert y.shape == (10, 50, 30, 40)

    y = transpose(t, '(b h w) c2->b c2 h w', b=30, h=10)
    print(y.shape)
    assert y.shape == (30, 50, 10, 40)


make_array_numpy = lambda *sizes: numpy.arange(numpy.prod(sizes)).reshape(sizes)
make_array_pytorch = lambda *sizes: torch.arange(int(numpy.prod(sizes))).reshape(sizes)
make_array_mxnetndarray = lambda *sizes: mxnet.ndarray.arange(int(numpy.prod(sizes))).reshape(sizes)
make_array_cupy = lambda *sizes: cupy.arange(int(numpy.prod(sizes))).reshape(sizes)
make_array_chainer = lambda *sizes: chainer.Variable(cupy.arange(int(numpy.prod(sizes)))).reshape(sizes)

test(make_array_numpy, transpose)
test(make_array_pytorch, transpose)
test(make_array_mxnetndarray, transpose)
test(make_array_cupy, transpose)
test(make_array_chainer, lambda *args, **kwargs: transpose(*args, **kwargs).data)


def check_tf():
    import tensorflow as tf

    def tf_transpose(tensor, *args, **kwargs):
        tensor_placeholder = tf.placeholder(dtype=tensor.dtype, shape=[None] * len(tensor.shape))
        result_variable = transpose(tensor_placeholder, *args, **kwargs)
        with tf.Session() as sess:
            result, = sess.run([result_variable], {tensor_placeholder: tensor})
        return result

    test(make_array_numpy, tf_transpose)


# check_tf()


def check_tf_eager():
    import tensorflow as tf
    tf.enable_eager_execution()

    def make_array_tf_eager(*sizes):
        return tf.reshape(tf.range(int(numpy.prod(sizes))), sizes)

    test(make_array_tf_eager, transpose)


check_tf_eager()
