"""Einops layers for Flax NNX. Existing Linen layers remain in :mod:`einops.layers.flax`."""

from typing import Any, cast

import jax
import jax.numpy as jnp
from flax import nnx

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin


class Rearrange(RearrangeMixin, nnx.Module):
    """Rearrange tensor axes inside an NNX model; axis lengths are keyword arguments."""

    def __call__(self, input: jax.Array) -> jax.Array:
        return self._apply_recipe(input)

    # The recipe mixin's pickle hooks omit NNX's graph metadata.
    def __getstate__(self) -> dict[str, Any]:
        return vars(self).copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)


class Reduce(ReduceMixin, nnx.Module):
    """Apply an einops reduction inside an NNX model."""

    def __call__(self, input: jax.Array) -> jax.Array:
        return self._apply_recipe(input)

    # Preserve NNX's graph metadata, including when nested in an EinMix.
    def __getstate__(self) -> dict[str, Any]:
        return vars(self).copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)


class EinMix(_EinmixMixin, nnx.Module):
    """Trainable tensor mixing for NNX, with parameters initialized at construction.

    Like other NNX layers, this takes ``rngs`` as a keyword-only argument.
    Axis lengths are passed as keywords, not in the Linen layer's ``sizes`` dict.
    The ``weight`` and optional ``bias`` are ``nnx.Param`` variables.

    Example:
        >>> layer = EinMix('b c -> b out', 'c out', 'out', c=4, out=8, rngs=nnx.Rngs(0))
        >>> layer(jnp.ones((2, 4))).shape
        (2, 8)
    """

    def __init__(
        self,
        pattern: str,
        weight_shape: str,
        bias_shape: str | None = None,
        *,
        rngs: nnx.Rngs,
        **axes_lengths: Any,
    ) -> None:
        # Initialization needs RNGs, but the deterministic layer has no RNG state to retain.
        self._rngs = rngs
        super().__init__(pattern, weight_shape, bias_shape, **axes_lengths)
        del self._rngs

    def _create_parameters(
        self, weight_shape: list[int], weight_bound: float, bias_shape: list[int] | None, bias_bound: float
    ) -> None:
        self.weight = nnx.Param(jax.nn.initializers.uniform(weight_bound)(self._rngs.params(), weight_shape))
        self.bias = (
            None
            if bias_shape is None
            else nnx.Param(jax.nn.initializers.uniform(bias_bound)(self._rngs.params(), bias_shape))
        )

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: str | None,
        pre_reshape_lengths: dict[str, int] | None,
        post_reshape_pattern: str | None,
        post_reshape_lengths: dict[str, int] | None,
    ) -> None:
        self.pre_rearrange = (
            None
            if pre_reshape_pattern is None
            else Rearrange(pre_reshape_pattern, **cast(dict[str, int], pre_reshape_lengths))
        )
        self.post_rearrange = (
            None
            if post_reshape_pattern is None
            else Rearrange(post_reshape_pattern, **cast(dict[str, int], post_reshape_lengths))
        )

    def __call__(self, input: jax.Array) -> jax.Array:
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = jnp.einsum(self.einsum_pattern, input, self.weight[...])
        if self.bias is not None:
            result += self.bias[...]
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
