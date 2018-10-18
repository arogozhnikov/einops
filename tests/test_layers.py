import pickle
import tempfile
from collections import namedtuple
import numpy

from einops import rearrange, reduce
from einops.einops import _reductions
from . import collect_test_backends

__author__ = 'Alex Rogozhnikov'

testcase = namedtuple('testcase', ['pattern', 'axes_lengths', 'input_shape', 'wrong_shapes'])

rearrangement_patterns = [
    testcase('b c h w -> b (c h w)', dict(c=20), (10, 20, 30, 40),
             [(), (10,), (10, 10, 10), (10, 21, 30, 40), [1, 20, 1, 1, 1]]),
    testcase('b c (h1 h2) (w1 w2) -> b (c h2 w2) h1 w1', dict(h2=2, w2=2), (10, 20, 30, 40),
             [(), (1, 1, 1, 1), (1, 10, 3), ()]),
    testcase('b ... c -> c b ...', dict(b=10), (10, 20, 30),
             [(), (10,), (5, 10)]),
]


def test_rearrange_imperative():
    for backend in collect_test_backends(symbolic=False, layers=True):
        print('Test layer for ', backend.framework_name)

        for pattern, axes_lengths, input_shape, wrong_shapes in rearrangement_patterns:
            x = numpy.arange(numpy.prod(input_shape), dtype='float32').reshape(input_shape)
            result_numpy = rearrange(x, pattern, **axes_lengths)
            layer = backend.layers().Rearrange(pattern, **axes_lengths)
            for shape in wrong_shapes:
                try:
                    layer(backend.from_numpy(numpy.zeros(shape, dtype='float32')))
                except:
                    pass
                else:
                    raise AssertionError('Failure expected')

            # simple pickling / unpickling
            layer2 = pickle.loads(pickle.dumps(layer))
            result1 = backend.to_numpy(layer(backend.from_numpy(x)))
            result2 = backend.to_numpy(layer2(backend.from_numpy(x)))
            assert numpy.allclose(result_numpy, result1)
            assert numpy.allclose(result1, result2)

            just_sum = backend.layers().Reduce('...->', reduction='sum')

            if 'mxnet' in backend.framework_name:
                with backend.mx.autograd.record():
                    variable = backend.from_numpy(x)
                    result = just_sum(layer(variable))
            else:
                variable = backend.from_numpy(x)
                result = just_sum(layer(variable))

            result.backward()
            assert numpy.allclose(backend.to_numpy(variable.grad), 1)


def test_rearrange_symbolic():
    for backend in collect_test_backends(symbolic=True, layers=True):
        print('Test layer for ', backend.framework_name)

        for pattern, axes_lengths, input_shape, wrong_shapes in rearrangement_patterns:
            x = numpy.arange(numpy.prod(input_shape), dtype='float32').reshape(input_shape)
            result_numpy = rearrange(x, pattern, **axes_lengths)
            layer = backend.layers().Rearrange(pattern, **axes_lengths)
            shapes = [input_shape]
            if 'mxnet' not in backend.framework_name:
                shapes.append([None] * len(input_shape))

            for shape in shapes:
                symbol = backend.create_symbol(shape)
                eval_inputs = [(symbol, x)]

                result_symbol1 = layer(symbol)
                result1 = backend.eval_symbol(result_symbol1, eval_inputs)
                assert numpy.allclose(result_numpy, result1)

                if 'keras' not in backend.framework_name:
                    # simple pickling / unpickling
                    # keras bug - fails for pickling
                    layer2 = pickle.loads(pickle.dumps(layer))
                    result_symbol2 = layer2(symbol)
                    result2 = backend.eval_symbol(result_symbol2, eval_inputs)
                    assert numpy.allclose(result1, result2)
                else:
                    import keras
                    import einops.layers.keras
                    model = keras.models.Model(symbol, result_symbol1)
                    result2 = model.predict_on_batch(x)

                    # create a temporary file using a context manager
                    with tempfile.NamedTemporaryFile(mode='r+b') as fp:
                        keras.models.save_model(model, fp.name)
                        model2 = keras.models.load_model(fp.name,
                                                         custom_objects=einops.layers.keras.keras_custom_objects)

                    result3 = model2.predict_on_batch(x)
                    assert numpy.allclose(result1, result2)
                    assert numpy.allclose(result1, result3)

                # now testing back-propagation
                just_sum = backend.layers().Reduce('...->', reduction='sum')

                result_sum1 = backend.eval_symbol(just_sum(result_symbol1), eval_inputs)
                result_sum2 = numpy.sum(x)

                assert numpy.allclose(result_sum1, result_sum2)


reduction_patterns = rearrangement_patterns + [
    testcase('b c h w -> b ()', dict(b=10), (10, 20, 30, 40),
             [(10,), (10, 20, 30)]),
    testcase('b c (h1 h2) (w1 w2) -> b c h1 w1', dict(h1=15, h2=2, w2=2), (10, 20, 30, 40),
             [(10, 20, 31, 40)]),
    testcase('b ... c -> b', dict(b=10), (10, 20, 30, 40),
             [(10,), (11, 10)]),
]


def test_reduce_imperative():
    for backend in collect_test_backends(symbolic=False, layers=True):
        print('Test layer for ', backend.framework_name)
        for reduction in _reductions:
            for pattern, axes_lengths, input_shape, wrong_shapes in reduction_patterns:
                print(backend, reduction, pattern, axes_lengths, input_shape, wrong_shapes)
                x = numpy.arange(1, 1 + numpy.prod(input_shape), dtype='float32').reshape(input_shape)
                x /= x.mean()
                result_numpy = reduce(x, pattern, reduction, **axes_lengths)
                layer = backend.layers().Reduce(pattern, reduction, **axes_lengths)
                for shape in wrong_shapes:
                    try:
                        layer(backend.from_numpy(numpy.zeros(shape, dtype='float32')))
                    except:
                        pass
                    else:
                        raise AssertionError('Failure expected')

                # simple pickling / unpickling
                layer2 = pickle.loads(pickle.dumps(layer))
                result1 = backend.to_numpy(layer(backend.from_numpy(x)))
                result2 = backend.to_numpy(layer2(backend.from_numpy(x)))
                assert numpy.allclose(result_numpy, result1)
                assert numpy.allclose(result1, result2)

                just_sum = backend.layers().Reduce('...->', reduction='sum')

                if 'mxnet' in backend.framework_name:
                    with backend.mx.autograd.record():
                        variable = backend.from_numpy(x)
                        result = just_sum(layer(variable))
                else:
                    variable = backend.from_numpy(x)
                    result = just_sum(layer(variable))

                result.backward()
                grad = backend.to_numpy(variable.grad)
                if reduction == 'sum':
                    assert numpy.allclose(grad, 1)
                if reduction == 'mean':
                    assert numpy.allclose(grad, grad.min())
                if reduction in ['max', 'min']:
                    assert numpy.all(numpy.in1d(grad, [0, 1]))
                    assert numpy.sum(grad) > 0.5


def test_reduce_symbolic():
    for backend in collect_test_backends(symbolic=True, layers=True):
        print('Test layer for ', backend.framework_name)
        for reduction in _reductions:
            for pattern, axes_lengths, input_shape, wrong_shapes in reduction_patterns:
                x = numpy.arange(1, 1 + numpy.prod(input_shape), dtype='float32').reshape(input_shape)
                x /= x.mean()
                result_numpy = reduce(x, pattern, reduction, **axes_lengths)
                layer = backend.layers().Reduce(pattern, reduction, **axes_lengths)
                shapes = [input_shape]
                if 'mxnet' not in backend.framework_name:
                    shapes.append([None] * len(input_shape))

                for shape in shapes:
                    symbol = backend.create_symbol(shape)
                    eval_inputs = [(symbol, x)]

                    result_symbol1 = layer(symbol)
                    result1 = backend.eval_symbol(result_symbol1, eval_inputs)
                    assert numpy.allclose(result_numpy, result1)

                    if 'keras' not in backend.framework_name:
                        # simple pickling / unpickling
                        # keras bug - fails for pickling
                        layer2 = pickle.loads(pickle.dumps(layer))
                        result_symbol2 = layer2(symbol)
                        result2 = backend.eval_symbol(result_symbol2, eval_inputs)
                        assert numpy.allclose(result1, result2)
                    else:
                        import keras
                        import einops.layers.keras
                        model = keras.models.Model(symbol, result_symbol1)
                        result2 = model.predict_on_batch(x)

                        # create a temporary file using a context manager
                        with tempfile.NamedTemporaryFile(mode='r+b') as fp:
                            keras.models.save_model(model, fp.name)
                            model2 = keras.models.load_model(fp.name,
                                                             custom_objects=einops.layers.keras.keras_custom_objects)

                        result3 = model2.predict_on_batch(x)
                        assert numpy.allclose(result1, result2)
                        assert numpy.allclose(result1, result3)
