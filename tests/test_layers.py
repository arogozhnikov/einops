import pickle
import tempfile

import numpy

from . import collect_test_settings

__author__ = 'Alex Rogozhnikov'

rearrangement_patterns = [
    ('b c h w -> b (c h w)', dict(b=10), (10, 20, 30, 40), (10, 20 * 30 * 40)),
    ('b c (h1 h2) (w1 w2) -> b (c h2 w2) h1 w1', dict(h1=15, h2=2, w2=2), (10, 20, 30, 40), (10, 20 * 2 * 2, 15, 20)),
    ('b ... c -> c b ...', dict(b=10, c=40), (10, 20, 30, 40), (40, 10, 20, 30)),
]


# reduction_patterns = rearrangement_patterns + [
#     ('b c h w -> b ()', dict(b=10), (10, 20, 30, 40), (10, 1)),
#     ('b c (h1 h2) (w1 w2) -> b c h1 w1', dict(h1=15, h2=2, w2=2), (10, 20, 30, 40), (10, 20, 15, 20)),
#     ('b ... c -> b', dict(b=10, c=40), (10, 20, 30, 40), (10,)),
# ]


def test_rearrange_imperative():
    for backend in collect_test_settings(symbolic=False, layers=True):
        print('Test layer for ', backend.framework_name)
        RearrangeLayer = backend.layers().Rearrange

        for pattern, axes_lengths, input_shape, result_shape in rearrangement_patterns:
            x = numpy.arange(numpy.prod(input_shape), dtype='float32').reshape(input_shape)
            layer = RearrangeLayer(pattern, **axes_lengths)
            assert layer(backend.from_numpy(x)).shape == result_shape
            for shape in [(), (10,), (10, 10, 10), (15, 20, 31, 40), (10, 1, 1, 1, 1)]:
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
    for backend in collect_test_settings(symbolic=True, layers=True):
        print('Test layer for ', backend.framework_name)
        RearrangeLayer = backend.layers().Rearrange

        for pattern, axes_lengths, input_shape, result_shape in rearrangement_patterns:
            x = numpy.arange(numpy.prod(input_shape), dtype='float32').reshape(input_shape)
            layer = RearrangeLayer(pattern, **axes_lengths)
            shapes = [input_shape]
            if 'mxnet' not in backend.framework_name:
                shapes.append([None] * len(input_shape))

            for shape in shapes:
                symbol = backend.create_symbol(shape)
                eval_inputs = [(symbol, x)]

                result_symbol1 = layer(symbol)
                result1 = backend.eval_symbol(result_symbol1, eval_inputs)
                assert result1.shape == result_shape

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
