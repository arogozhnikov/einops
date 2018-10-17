import einops.layers.chainer
import einops.layers.gluon
import einops.layers.torch
import einops.layers.keras

__author__ = 'Alex Rogozhnikov'

import pickle
from copy import deepcopy

import chainer
import mxnet
import numpy
import torch
import keras

import tempfile
from einops import backends

rearrangement_patterns = [
    ('b c h w -> b (c h w)', dict(b=10), (10, 20, 30, 40), (10, 20 * 30 * 40)),
    ('b c (h1 h2) (w1 w2) -> b (c h2 w2) h1 w1', dict(h1=15, h2=2, w2=2), (10, 20, 30, 40), (10, 20 * 2 * 2, 15, 20)),
    # ('b ... c -> c b ...', dict(b=10, c=40), (10, 20, 30, 40), (40, 10, 20, 30)),
]


# reduction_patterns = rearrangement_patterns + [
#     ('b c h w -> b ()', dict(b=10), (10, 20, 30, 40), (10, 1)),
#     ('b c (h1 h2) (w1 w2) -> b c h1 w1', dict(h1=15, h2=2, w2=2), (10, 20, 30, 40), (10, 20, 15, 20)),
#     ('b ... c -> b', dict(b=10, c=40), (10, 20, 30, 40), (10,)),
# ]


def test_keras():
    for pattern, axes_lengths, input_shape, result_shape in rearrangement_patterns:
        x = numpy.arange(numpy.prod(input_shape), dtype='float32').reshape(input_shape)

        keras_input = keras.layers.Input(shape=input_shape[1:])
        layer = einops.layers.keras.Rearrange(pattern, **axes_lengths)
        output = layer(keras_input)
        # output = keras.layers.Activation('tanh')(keras_input)
        model = keras.models.Model(keras_input, output)
        result1 = model.predict_on_batch(x)
        assert result1.shape == result_shape

        # create a temporary file using a context manager
        with tempfile.NamedTemporaryFile(mode='r+b') as fp:
            keras.models.save_model(model, fp.name)
            model2 = keras.models.load_model(fp.name, custom_objects=einops.layers.keras.keras_custom_objects)

        result2 = model2.predict_on_batch(x)
        assert numpy.allclose(result1, result2)

        # model3 = pickle.loads(pickle.dumps(model))
        # result3 = model3.predict_on_batch(x)
        # assert numpy.allclose(result1, result3)

    print('Tested keras layer')


test_keras()


def test_rearrange():
    backend_pairs = [
        (backends.TorchBackend(), einops.layers.torch.Rearrange),
        (backends.ChainerBackend(), einops.layers.chainer.Rearrange),
        (backends.GluonBackend(), einops.layers.gluon.Rearrange),
    ]

    for backend, RearrangeLayer in backend_pairs:
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

            if RearrangeLayer == einops.layers.torch.Rearrange:
                layer3 = deepcopy(layer2)
            elif RearrangeLayer == einops.layers.chainer.Rearrange:
                layer3 = deepcopy(layer2)
            elif RearrangeLayer == einops.layers.gluon.Rearrange:
                # hybridization doesn't work
                # layer3 = layer2.hybridize()
                layer3 = deepcopy(layer2)
            result3 = backend.to_numpy(layer3(backend.from_numpy(x)))
            assert numpy.allclose(result1, result3)

            v = backend.from_numpy(x)
            if isinstance(v, torch.Tensor):
                v.requires_grad = True
            if isinstance(v, mxnet.nd.NDArray):
                v.attach_grad()

            if isinstance(v, chainer.Variable):
                chainer.functions.sum(layer(v)).backward()
            elif isinstance(v, mxnet.nd.NDArray):
                from mxnet import autograd
                with autograd.record():
                    layer(v).sum().backward()
            else:
                layer(v).sum().backward()

            assert numpy.allclose(backend.to_numpy(v.grad), 1)

        print('Tested layer for ', backend.framework_name)


test_rearrange()
