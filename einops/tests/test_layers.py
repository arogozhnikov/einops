import pickle
from collections import namedtuple

import numpy
import pytest

from einops import rearrange, reduce, EinopsError
from einops.tests import collect_test_backends, is_backend_tested, FLOAT_REDUCTIONS as REDUCTIONS

__author__ = "Alex Rogozhnikov"

testcase = namedtuple("testcase", ["pattern", "axes_lengths", "input_shape", "wrong_shapes"])

rearrangement_patterns = [
    testcase(
        "b c h w -> b (c h w)",
        dict(c=20),
        (10, 20, 30, 40),
        [(), (10,), (10, 10, 10), (10, 21, 30, 40), [1, 20, 1, 1, 1]],
    ),
    testcase(
        "b c (h1 h2) (w1 w2) -> b (c h2 w2) h1 w1",
        dict(h2=2, w2=2),
        (10, 20, 30, 40),
        [(), (1, 1, 1, 1), (1, 10, 3), ()],
    ),
    testcase(
        "b ... c -> c b ...",
        dict(b=10),
        (10, 20, 30),
        [(), (10,), (5, 10)],
    ),
]


def test_rearrange_imperative():
    for backend in collect_test_backends(symbolic=False, layers=True):
        print("Test layer for ", backend.framework_name)

        for pattern, axes_lengths, input_shape, wrong_shapes in rearrangement_patterns:
            x = numpy.arange(numpy.prod(input_shape), dtype="float32").reshape(input_shape)
            result_numpy = rearrange(x, pattern, **axes_lengths)
            layer = backend.layers().Rearrange(pattern, **axes_lengths)
            for shape in wrong_shapes:
                try:
                    layer(backend.from_numpy(numpy.zeros(shape, dtype="float32")))
                except BaseException:
                    pass
                else:
                    raise AssertionError("Failure expected")

            # simple pickling / unpickling
            layer2 = pickle.loads(pickle.dumps(layer))
            result1 = backend.to_numpy(layer(backend.from_numpy(x)))
            result2 = backend.to_numpy(layer2(backend.from_numpy(x)))
            assert numpy.allclose(result_numpy, result1)
            assert numpy.allclose(result1, result2)

            just_sum = backend.layers().Reduce("...->", reduction="sum")

            variable = backend.from_numpy(x)
            result = just_sum(layer(variable))

            result.backward()
            assert numpy.allclose(backend.to_numpy(variable.grad), 1)


def test_rearrange_symbolic():
    for backend in collect_test_backends(symbolic=True, layers=True):
        print("Test layer for ", backend.framework_name)

        for pattern, axes_lengths, input_shape, wrong_shapes in rearrangement_patterns:
            x = numpy.arange(numpy.prod(input_shape), dtype="float32").reshape(input_shape)
            result_numpy = rearrange(x, pattern, **axes_lengths)
            layer = backend.layers().Rearrange(pattern, **axes_lengths)
            input_shape_of_nones = [None] * len(input_shape)
            shapes = [input_shape, input_shape_of_nones]

            for shape in shapes:
                symbol = backend.create_symbol(shape)
                eval_inputs = [(symbol, x)]

                result_symbol1 = layer(symbol)
                result1 = backend.eval_symbol(result_symbol1, eval_inputs)
                assert numpy.allclose(result_numpy, result1)

                layer2 = pickle.loads(pickle.dumps(layer))
                result_symbol2 = layer2(symbol)
                result2 = backend.eval_symbol(result_symbol2, eval_inputs)
                assert numpy.allclose(result1, result2)

                # now testing back-propagation
                just_sum = backend.layers().Reduce("...->", reduction="sum")

                result_sum1 = backend.eval_symbol(just_sum(result_symbol1), eval_inputs)
                result_sum2 = numpy.sum(x)

                assert numpy.allclose(result_sum1, result_sum2)


reduction_patterns = rearrangement_patterns + [
    testcase("b c h w -> b ()", dict(b=10), (10, 20, 30, 40), [(10,), (10, 20, 30)]),
    testcase("b c (h1 h2) (w1 w2) -> b c h1 w1", dict(h1=15, h2=2, w2=2), (10, 20, 30, 40), [(10, 20, 31, 40)]),
    testcase("b ... c -> b", dict(b=10), (10, 20, 30, 40), [(10,), (11, 10)]),
]


def test_reduce_imperative():
    for backend in collect_test_backends(symbolic=False, layers=True):
        print("Test layer for ", backend.framework_name)
        for reduction in REDUCTIONS:
            for pattern, axes_lengths, input_shape, wrong_shapes in reduction_patterns:
                print(backend, reduction, pattern, axes_lengths, input_shape, wrong_shapes)
                x = numpy.arange(1, 1 + numpy.prod(input_shape), dtype="float32").reshape(input_shape)
                x /= x.mean()
                result_numpy = reduce(x, pattern, reduction, **axes_lengths)
                layer = backend.layers().Reduce(pattern, reduction, **axes_lengths)
                for shape in wrong_shapes:
                    try:
                        layer(backend.from_numpy(numpy.zeros(shape, dtype="float32")))
                    except BaseException:
                        pass
                    else:
                        raise AssertionError("Failure expected")

                # simple pickling / unpickling
                layer2 = pickle.loads(pickle.dumps(layer))
                result1 = backend.to_numpy(layer(backend.from_numpy(x)))
                result2 = backend.to_numpy(layer2(backend.from_numpy(x)))
                assert numpy.allclose(result_numpy, result1)
                assert numpy.allclose(result1, result2)

                just_sum = backend.layers().Reduce("...->", reduction="sum")

                variable = backend.from_numpy(x)
                result = just_sum(layer(variable))

                result.backward()
                grad = backend.to_numpy(variable.grad)
                if reduction == "sum":
                    assert numpy.allclose(grad, 1)
                if reduction == "mean":
                    assert numpy.allclose(grad, grad.min())
                if reduction in ["max", "min"]:
                    assert numpy.all(numpy.in1d(grad, [0, 1]))
                    assert numpy.sum(grad) > 0.5


def test_reduce_symbolic():
    for backend in collect_test_backends(symbolic=True, layers=True):
        print("Test layer for ", backend.framework_name)
        for reduction in REDUCTIONS:
            for pattern, axes_lengths, input_shape, wrong_shapes in reduction_patterns:
                x = numpy.arange(1, 1 + numpy.prod(input_shape), dtype="float32").reshape(input_shape)
                x /= x.mean()
                result_numpy = reduce(x, pattern, reduction, **axes_lengths)
                layer = backend.layers().Reduce(pattern, reduction, **axes_lengths)
                input_shape_of_nones = [None] * len(input_shape)
                shapes = [input_shape, input_shape_of_nones]

                for shape in shapes:
                    symbol = backend.create_symbol(shape)
                    eval_inputs = [(symbol, x)]

                    result_symbol1 = layer(symbol)
                    result1 = backend.eval_symbol(result_symbol1, eval_inputs)
                    assert numpy.allclose(result_numpy, result1)

                    layer2 = pickle.loads(pickle.dumps(layer))
                    result_symbol2 = layer2(symbol)
                    result2 = backend.eval_symbol(result_symbol2, eval_inputs)
                    assert numpy.allclose(result1, result2)


def create_torch_model(use_reduce=False, add_scripted_layer=False):
    if not is_backend_tested("torch"):
        pytest.skip()
    else:
        from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
        from einops.layers.torch import Rearrange, Reduce, EinMix
        import torch.jit

        return Sequential(
            Conv2d(3, 6, kernel_size=(5, 5)),
            Reduce("b c (h h2) (w w2) -> b c h w", "max", h2=2, w2=2) if use_reduce else MaxPool2d(kernel_size=2),
            Conv2d(6, 16, kernel_size=(5, 5)),
            Reduce("b c (h h2) (w w2) -> b c h w", "max", h2=2, w2=2),
            torch.jit.script(Rearrange("b c h w -> b (c h w)"))
            if add_scripted_layer
            else Rearrange("b c h w -> b (c h w)"),
            Linear(16 * 5 * 5, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            EinMix("b c1 -> (b c2)", weight_shape="c1 c2", bias_shape="c2", c1=84, c2=84),
            EinMix("(b c2) -> b c3", weight_shape="c2 c3", bias_shape="c3", c2=84, c3=84),
            Linear(84, 10),
        )


def test_torch_layer():
    if not is_backend_tested("torch"):
        pytest.skip()
    else:
        # checked that torch present
        import torch
        import torch.jit

        model1 = create_torch_model(use_reduce=True)
        model2 = create_torch_model(use_reduce=False)
        input = torch.randn([10, 3, 32, 32])
        # random models have different predictions
        assert not torch.allclose(model1(input), model2(input))
        model2.load_state_dict(pickle.loads(pickle.dumps(model1.state_dict())))
        assert torch.allclose(model1(input), model2(input))

        # tracing (freezing)
        model3 = torch.jit.trace(model2, example_inputs=input)
        torch.testing.assert_close(model1(input), model3(input), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(model1(input + 1), model3(input + 1), atol=1e-3, rtol=1e-3)

        model4 = torch.jit.trace(model2, example_inputs=input)
        torch.testing.assert_close(model1(input), model4(input), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(model1(input + 1), model4(input + 1), atol=1e-3, rtol=1e-3)


def test_torch_layers_scripting():
    if not is_backend_tested("torch"):
        pytest.skip()
    else:
        import torch

        for script_layer in [False, True]:
            model1 = create_torch_model(use_reduce=True, add_scripted_layer=script_layer)
            model2 = torch.jit.script(model1)
            input = torch.randn([10, 3, 32, 32])

            torch.testing.assert_close(model1(input), model2(input), atol=1e-3, rtol=1e-3)


def test_keras_layer():
    if not is_backend_tested("tensorflow"):
        pytest.skip()
    else:
        import tensorflow as tf

        if tf.__version__ < "2.16.":
            # current implementation of layers follows new TF interface
            pytest.skip()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D as Conv2d, Dense as Linear, ReLU
        from einops.layers.keras import Rearrange, Reduce, EinMix, keras_custom_objects

        def create_keras_model():
            return Sequential(
                [
                    Conv2d(6, kernel_size=5, input_shape=[32, 32, 3]),
                    Reduce("b c (h h2) (w w2) -> b c h w", "max", h2=2, w2=2),
                    Conv2d(16, kernel_size=5),
                    Reduce("b c (h h2) (w w2) -> b c h w", "max", h2=2, w2=2),
                    Rearrange("b c h w -> b (c h w)"),
                    Linear(120),
                    ReLU(),
                    Linear(84),
                    ReLU(),
                    EinMix("b c1 -> (b c2)", weight_shape="c1 c2", bias_shape="c2", c1=84, c2=84),
                    EinMix("(b c2) -> b c3", weight_shape="c2 c3", bias_shape="c3", c2=84, c3=84),
                    Linear(10),
                ]
            )

        model1 = create_keras_model()
        model2 = create_keras_model()

        input = numpy.random.normal(size=[10, 32, 32, 3]).astype("float32")
        # two randomly init models should provide different outputs
        assert not numpy.allclose(model1.predict_on_batch(input), model2.predict_on_batch(input))

        # get some temp filename
        tmp_model_filename = "/tmp/einops_tf_model.h5"
        # save arch + weights
        print("temp_path_keras1", tmp_model_filename)
        tf.keras.models.save_model(model1, tmp_model_filename)
        model3 = tf.keras.models.load_model(tmp_model_filename, custom_objects=keras_custom_objects)

        numpy.testing.assert_allclose(model1.predict_on_batch(input), model3.predict_on_batch(input))

        weight_filename = "/tmp/einops_tf_model.weights.h5"
        # save arch as json
        model4 = tf.keras.models.model_from_json(model1.to_json(), custom_objects=keras_custom_objects)
        model1.save_weights(weight_filename)
        model4.load_weights(weight_filename)
        model2.load_weights(weight_filename)
        # check that differently-inialized model receives same weights
        numpy.testing.assert_allclose(model1.predict_on_batch(input), model2.predict_on_batch(input))
        # ulimate test
        # save-load architecture, and then load weights - should return same result
        numpy.testing.assert_allclose(model1.predict_on_batch(input), model4.predict_on_batch(input))


def test_flax_layers():
    """
    One-off simple tests for Flax layers.
    Unfortunately, Flax layers have a different interface from other layers.
    """
    if not is_backend_tested("jax"):
        pytest.skip()
    else:
        import jax
        import jax.numpy as jnp

        import flax
        from flax import linen as nn
        from einops.layers.flax import EinMix, Reduce, Rearrange

        class NN(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = EinMix(
                    "b (h h2) (w w2) c -> b h w c_out", "h2 w2 c c_out", "c_out", sizes=dict(h2=2, w2=3, c=4, c_out=5)
                )(x)
                x = Rearrange("b h w c -> b (w h c)", sizes=dict(c=5))(x)
                x = Reduce("b hwc -> b", "mean", dict(hwc=2 * 3 * 5))(x)
                return x

        model = NN()
        fixed_input = jnp.ones([10, 2 * 2, 3 * 3, 4])
        params = model.init(jax.random.PRNGKey(0), fixed_input)

        def eval_at_point(params):
            return jnp.linalg.norm(model.apply(params, fixed_input))

        vandg = jax.value_and_grad(eval_at_point)
        value0 = eval_at_point(params)
        value1, grad1 = vandg(params)
        assert jnp.allclose(value0, value1)

        params2 = jax.tree_map(lambda x1, x2: x1 - x2 * 0.001, params, grad1)

        value2 = eval_at_point(params2)
        assert value0 >= value2, (value0, value2)

        # check serialization
        fbytes = flax.serialization.to_bytes(params)
        _loaded = flax.serialization.from_bytes(params, fbytes)


def test_einmix_decomposition():
    """
    Testing that einmix correctly decomposes into smaller transformations.
    """
    from einops.layers._einmix import _EinmixDebugger

    mixin1 = _EinmixDebugger(
        "a b c d e -> e d c b a",
        weight_shape="d a b",
        d=2, a=3, b=5,
    )  # fmt: off
    assert mixin1.pre_reshape_pattern is None
    assert mixin1.post_reshape_pattern is None
    assert mixin1.einsum_pattern == "abcde,dab->edcba"
    assert mixin1.saved_weight_shape == [2, 3, 5]
    assert mixin1.saved_bias_shape is None

    mixin2 = _EinmixDebugger(
        "a b c d e -> e d c b a",
        weight_shape="d a b",
        bias_shape="a b c d e",
        a=1, b=2, c=3, d=4, e=5,
    )  # fmt: off
    assert mixin2.pre_reshape_pattern is None
    assert mixin2.post_reshape_pattern is None
    assert mixin2.einsum_pattern == "abcde,dab->edcba"
    assert mixin2.saved_weight_shape == [4, 1, 2]
    assert mixin2.saved_bias_shape == [5, 4, 3, 2, 1]

    mixin3 = _EinmixDebugger(
        "... -> ...",
        weight_shape="",
        bias_shape="",
    )  # fmt: off
    assert mixin3.pre_reshape_pattern is None
    assert mixin3.post_reshape_pattern is None
    assert mixin3.einsum_pattern == "...,->..."
    assert mixin3.saved_weight_shape == []
    assert mixin3.saved_bias_shape == []

    mixin4 = _EinmixDebugger(
        "b a ...  -> b c ...",
        weight_shape="b a c",
        a=1, b=2, c=3,
    )  # fmt: off
    assert mixin4.pre_reshape_pattern is None
    assert mixin4.post_reshape_pattern is None
    assert mixin4.einsum_pattern == "ba...,bac->bc..."
    assert mixin4.saved_weight_shape == [2, 1, 3]
    assert mixin4.saved_bias_shape is None

    mixin5 = _EinmixDebugger(
        "(b a) ... -> b c (...)",
        weight_shape="b a c",
        a=1, b=2, c=3,
    )  # fmt: off
    assert mixin5.pre_reshape_pattern == "(b a) ... -> b a ..."
    assert mixin5.pre_reshape_lengths == dict(a=1, b=2)
    assert mixin5.post_reshape_pattern == "b c ... -> b c (...)"
    assert mixin5.einsum_pattern == "ba...,bac->bc..."
    assert mixin5.saved_weight_shape == [2, 1, 3]
    assert mixin5.saved_bias_shape is None

    mixin6 = _EinmixDebugger(
        "b ... (a c) -> b ... (a d)",
        weight_shape="c d",
        bias_shape="a d",
        a=1, c=3, d=4,
    )  # fmt: off
    assert mixin6.pre_reshape_pattern == "b ... (a c) -> b ... a c"
    assert mixin6.pre_reshape_lengths == dict(a=1, c=3)
    assert mixin6.post_reshape_pattern == "b ... a d -> b ... (a d)"
    assert mixin6.einsum_pattern == "b...ac,cd->b...ad"
    assert mixin6.saved_weight_shape == [3, 4]
    assert mixin6.saved_bias_shape == [1, 1, 4]  # (b) a d, ellipsis does not participate

    mixin7 = _EinmixDebugger(
        "a ... (b c) -> a (... d b)",
        weight_shape="c d b",
        bias_shape="d b",
        b=2, c=3, d=4,
    )  # fmt: off
    assert mixin7.pre_reshape_pattern == "a ... (b c) -> a ... b c"
    assert mixin7.pre_reshape_lengths == dict(b=2, c=3)
    assert mixin7.post_reshape_pattern == "a ... d b -> a (... d b)"
    assert mixin7.einsum_pattern == "a...bc,cdb->a...db"
    assert mixin7.saved_weight_shape == [3, 4, 2]
    assert mixin7.saved_bias_shape == [1, 4, 2]  # (a) d b, ellipsis does not participate


def test_einmix_restrictions():
    """
    Testing different cases
    """
    from einops.layers._einmix import _EinmixDebugger

    with pytest.raises(EinopsError):
        _EinmixDebugger(
            "a b c d e -> e d c b a",
            weight_shape="d a b",
            d=2, a=3, # missing b
        )  # fmt: off

    with pytest.raises(EinopsError):
        _EinmixDebugger(
            "a b c d e -> e d c b a",
            weight_shape="w a b",
            d=2, a=3, b=1 # missing d
        )  # fmt: off

    with pytest.raises(EinopsError):
        _EinmixDebugger(
            "(...) a -> ... a",
            weight_shape="a", a=1, # ellipsis on the left
        )  # fmt: off

    with pytest.raises(EinopsError):
        _EinmixDebugger(
            "(...) a -> a ...",
            weight_shape="a", a=1, # ellipsis on the right side after bias axis
            bias_shape='a',
        )  # fmt: off
