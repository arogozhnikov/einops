import numpy as np
import pytest

from einops import _backends, asnumpy, einsum, pack, parse_shape, rearrange, reduce, repeat
from einops.tests import is_backend_tested


@pytest.mark.parametrize("operation", ["min", "max", "sum", "mean", "prod", "any", "all"])
def test_numpy_reduction_scalar_can_be_reused(operation):
    array = np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    scalar = reduce(array, "a b ->", operation)
    expected = getattr(array, operation)()

    # Keep NumPy's scalar return type, including bool scalars from any/all.
    assert isinstance(scalar, np.generic)
    assert type(scalar) is type(expected)
    assert scalar == expected
    assert parse_shape(scalar, "") == {}
    assert asnumpy(scalar) is scalar
    np.testing.assert_array_equal(rearrange(scalar, "-> 1"), expected.reshape(1))
    np.testing.assert_array_equal(repeat(scalar, "-> copies", copies=3), np.repeat(expected, 3))


@pytest.mark.parametrize(
    "scalar",
    [np.int8(-3), np.int64(2**40 + 1), np.uint64(2**64 - 1), np.float32(1.25), np.complex128(2 + 3j), np.bool_(True)],
)
def test_numpy_scalar_shape_operations_preserve_dtype(scalar):
    result = rearrange(scalar, "-> 1 1")
    assert type(result) is np.ndarray
    assert result.dtype == scalar.dtype
    np.testing.assert_array_equal(result, np.asarray(scalar).reshape(1, 1))
    assert parse_shape(scalar, "...") == {}


def test_numpy_scalars_can_be_stacked_packed_and_contracted():
    first = reduce(np.array([2, 3]), "i ->", "sum")
    second = reduce(np.array([4, 5]), "i ->", "sum")

    np.testing.assert_array_equal(rearrange([first, second], "i -> i 1"), [[5], [9]])
    packed, shapes = pack([first, np.array([second])], "*")
    np.testing.assert_array_equal(packed, [5, 9])
    assert shapes == [(), (1,)]
    assert einsum(first, second, ", ->") == 45


@pytest.mark.parametrize("array", [np.array(5), np.ma.array([1, 2], mask=[False, True]), np.ma.masked])
def test_numpy_array_subclasses_keep_their_backend(array):
    assert _backends.get_backend(array).framework_name == "numpy"
    reshaped = rearrange(array, "... -> 1 ...")
    expected = np.expand_dims(array, 0)
    assert type(reshaped) is type(expected)
    np.testing.assert_array_equal(reshaped, expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(reshaped), np.expand_dims(np.ma.getmaskarray(array), 0))


@pytest.mark.parametrize("value", [1, 1.25, True])
def test_python_scalars_are_not_numpy_tensors(value):
    with pytest.raises(RuntimeError, match="Tensor type unknown"):
        rearrange(value, "-> 1")


@pytest.mark.skipif(not is_backend_tested("jax"), reason="JAX backend is not selected")
@pytest.mark.parametrize("backend_order", [("numpy", "jax"), ("jax", "numpy")])
def test_numpy_scalar_dispatch_with_jax_loaded(monkeypatch, backend_order):
    import jax
    import jax.numpy as jnp

    backends = {"numpy": _backends.NumpyBackend(), "jax": _backends.JaxBackend()}
    monkeypatch.setattr(_backends, "_loaded_backends", {name: backends[name] for name in backend_order})
    monkeypatch.setattr(_backends, "_type2backend", {})

    scalar = np.int64(2**40 + 1)
    assert not backends["jax"].is_appropriate_type(scalar)
    assert _backends.get_backend(scalar) is backends["numpy"]
    reshaped = rearrange(scalar, "-> 1")
    assert type(reshaped) is np.ndarray
    assert reshaped.dtype == scalar.dtype
    np.testing.assert_array_equal(reshaped, [2**40 + 1])

    array = jnp.arange(3)
    assert _backends.get_backend(array) is backends["jax"]
    result = jax.jit(lambda x: rearrange(reduce(x, "i ->", "sum"), "-> 1"))(array)
    assert isinstance(result, jax.Array)
    np.testing.assert_array_equal(result, [3])
