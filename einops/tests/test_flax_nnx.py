"""Flax NNX layers share the existing recipes without changing the Linen layers."""

from typing import cast

import numpy as np
import pytest

from einops.tests import is_backend_tested


@pytest.fixture
def nnx_modules():
    if not is_backend_tested("jax"):
        pytest.skip()
    from flax import nnx

    from einops.layers.flax_nnx import EinMix, Rearrange, Reduce

    return nnx, EinMix, Rearrange, Reduce


@pytest.mark.parametrize("shape", [(2, 3), (2, 4, 3), (2, 1, 4, 3)])
def test_nnx_rearrange_and_reduce(nnx_modules, shape):
    import jax.numpy as jnp

    from einops import rearrange, reduce

    nnx, _, Rearrange, Reduce = nnx_modules
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    rearranger = Rearrange("b ... c -> c ... b", c=3)
    reducer = Reduce("b ... c -> b c", "mean", c=3)
    for layer, expected in [
        (rearranger, rearrange(x, "b ... c -> c ... b", c=3)),
        (reducer, reduce(x, "b ... c -> b c", "mean", c=3)),
    ]:
        np.testing.assert_array_equal(layer(x), expected)
        np.testing.assert_array_equal(nnx.jit(lambda m, a: m(a))(layer, x), expected)
        assert not nnx.state(layer)


@pytest.mark.parametrize("bias_shape", [None, "out"])
@pytest.mark.parametrize(
    ("pattern", "shape", "expected_shape"),
    [
        ("b c -> b out", (2, 4), (2, 5)),
        ("... c -> ... out", (2, 3, 4), (2, 3, 5)),
        ("b (h c) -> b h out", (2, 12), (2, 3, 5)),
        ("b c -> (b out)", (2, 4), (10,)),
    ],
)
def test_nnx_einmix_matches_linen(nnx_modules, pattern, shape, expected_shape, bias_shape):
    import jax
    import jax.numpy as jnp

    from einops.layers.flax import EinMix as LinenEinMix

    nnx, EinMix, _, _ = nnx_modules
    sizes = dict(c=4, out=5)
    layer = EinMix(pattern, "c out", bias_shape, rngs=nnx.Rngs(0), **sizes)
    linen = LinenEinMix(pattern, "c out", bias_shape, sizes=sizes)
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 10
    params = {"weight": layer.weight[...]}
    if bias_shape is not None:
        params["bias"] = layer.bias[...]
    expected = cast(jax.Array, linen.apply({"params": params}, x))
    assert expected.shape == expected_shape
    np.testing.assert_allclose(layer(x), expected)
    np.testing.assert_allclose(nnx.jit(lambda m, a: m(a))(layer, x), expected)

    # Gradients must also agree when passing parameters through the existing Linen layer.
    def linen_loss(params):
        return jnp.square(cast(jax.Array, linen.apply({"params": params}, x))).mean()

    grads = nnx.to_pure_dict(nnx.grad(lambda m: jnp.square(m(x)).mean())(layer))
    expected_grads = jax.grad(linen_loss)(params)
    np.testing.assert_allclose(grads["weight"], expected_grads["weight"])
    if bias_shape is not None:
        np.testing.assert_allclose(grads["bias"], expected_grads["bias"])
    else:
        assert layer.bias is None
        assert set(grads) == {"weight"}


def test_nnx_training_and_state_round_trip(nnx_modules):
    import jax
    import jax.numpy as jnp

    nnx, EinMix, Rearrange, Reduce = nnx_modules
    model = nnx.Sequential(
        EinMix("b (h h2) (w w2) c -> b h w out", "h2 w2 c out", "out", rngs=nnx.Rngs(0), h2=2, w2=3, c=4, out=5),
        Rearrange("b h w c -> b (w h c)"),
        Reduce("b features -> b", "mean"),
    )
    x = jnp.ones((10, 4, 9, 4))

    @nnx.jit
    def train_step(model, x):
        loss, grads = nnx.value_and_grad(lambda m: jnp.square(m(x)).mean())(model)
        params = nnx.state(model, nnx.Param)
        nnx.update(model, jax.tree.map(lambda p, g: p - 0.01 * g, params, grads))
        return loss

    initial = train_step(model, x)
    assert jnp.square(model(x)).mean() < initial
    graph, state = nnx.split(model)
    restored = nnx.merge(graph, state)
    np.testing.assert_array_equal(restored(x), model(x))
    assert set(nnx.state(model, nnx.Param)["layers"][0]) == {"weight", "bias"}


def test_nnx_initialization_uses_params_stream(nnx_modules):
    import jax.numpy as jnp

    nnx, EinMix, _, _ = nnx_modules

    def make(rngs):
        return EinMix("b c -> b out", "c out", "out", rngs=rngs, c=4, out=5)

    rngs = nnx.Rngs(params=0)
    first, second = make(rngs), make(rngs)
    replay = make(nnx.Rngs(params=0))
    np.testing.assert_array_equal(first.weight[...], replay.weight[...])
    np.testing.assert_array_equal(first.bias[...], replay.bias[...])
    assert not jnp.array_equal(first.weight[...], second.weight[...])
    # RNG streams are consumed only at initialization, not retained as model state.
    assert set(nnx.state(first)) == {"weight", "bias"}


def test_nnx_einmix_invalid_weight_axes(nnx_modules):
    from einops import EinopsError

    nnx, EinMix, _, _ = nnx_modules
    with pytest.raises(EinopsError, match="Dimension out of weight should be specified"):
        EinMix("b c -> b out", "c out", rngs=nnx.Rngs(0), c=4)


def test_nnx_rearrange_reuses_rank_recipes(nnx_modules):
    import jax.numpy as jnp

    from einops import rearrange

    nnx, _, Rearrange, _ = nnx_modules
    layer = Rearrange("b ... c -> c ... b", c=3)
    apply = nnx.jit(lambda m, x: m(x))
    for shape in [(2, 3), (2, 4, 3), (2, 1, 4, 3)]:
        x = jnp.arange(np.prod(shape)).reshape(shape)
        np.testing.assert_array_equal(apply(layer, x), rearrange(x, "b ... c -> c ... b", c=3))


def test_nnx_patch_mixing_matches_explicit_einsum(nnx_modules):
    import jax.numpy as jnp

    nnx, EinMix, _, _ = nnx_modules
    layer = EinMix(
        "b (h h2) (w w2) c -> b h (w out)",
        "h2 w2 c out",
        "out",
        rngs=nnx.Rngs(0),
        h2=2,
        w2=3,
        c=4,
        out=5,
    )
    x = jnp.arange(2 * 4 * 9 * 4, dtype=jnp.float32).reshape(2, 4, 9, 4) / 100
    patches = x.reshape(2, 2, 2, 3, 3, 4)
    expected = jnp.einsum("bhpwqc,pqco->bhwo", patches, layer.weight[...])
    expected = expected + layer.bias[...]
    expected = expected.reshape(2, 2, 15)
    np.testing.assert_allclose(layer(x), expected, rtol=1e-6)
    np.testing.assert_allclose(nnx.jit(lambda m, a: m(a))(layer, x), expected, rtol=1e-6)


@pytest.mark.parametrize("kind", ["rearrange", "reduce", "einmix"])
def test_nnx_pickle_round_trip(nnx_modules, kind):
    import pickle

    import jax.numpy as jnp

    nnx, EinMix, Rearrange, Reduce = nnx_modules
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)
    if kind == "rearrange":
        layer = Rearrange("b (h c) -> c b h", c=4)
    elif kind == "reduce":
        layer = Reduce("b (h c) -> b c", "sum", c=4)
    else:
        # Both pre- and post-rearrange submodules must keep their NNX metadata too.
        layer = EinMix("b (h c) -> (b h out)", "c out", "out", rngs=nnx.Rngs(0), c=4, out=5)
    restored = pickle.loads(pickle.dumps(layer))
    np.testing.assert_array_equal(restored(x), layer(x))
    np.testing.assert_allclose(nnx.jit(lambda m, a: m(a))(restored, x), layer(x), rtol=1e-6)
    graph, state = nnx.split(restored)
    np.testing.assert_array_equal(nnx.merge(graph, state)(x), layer(x))
    if kind == "einmix":
        assert isinstance(restored.weight, nnx.Param)
        assert isinstance(restored.bias, nnx.Param)
        grads = nnx.grad(lambda m: jnp.square(m(x)).mean())(restored)
        assert set(grads) == {"weight", "bias"}
