from typing import cast

import torch

from einops._torch_specific import apply_for_scriptable_torch

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = "Alex Rogozhnikov"


class Rearrange(RearrangeMixin, torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        recipe = self._multirecipe[input.ndim]
        return apply_for_scriptable_torch(recipe, input, reduction_type="rearrange", axes_dims=self._axes_lengths)  # type: ignore[arg-type]

    def _apply_recipe(self, x):
        # overriding parent method to prevent its scripting
        pass


class Reduce(ReduceMixin, torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        recipe = self._multirecipe[input.ndim]
        return apply_for_scriptable_torch(recipe, input, reduction_type=self.reduction, axes_dims=self._axes_lengths)  # type: ignore[arg-type]

    def _apply_recipe(self, x):
        # overriding parent method to prevent its scripting
        pass


class EinMix(_EinmixMixin, torch.nn.Module):
    def _create_parameters(
        self, weight_shape: list[int], weight_bound: float, bias_shape: list[int] | None, bias_bound: float
    ) -> None:
        self.weight = torch.nn.Parameter(
            torch.zeros(weight_shape).uniform_(-weight_bound, weight_bound), requires_grad=True
        )

        self.bias: torch.nn.Parameter | None
        if bias_shape is not None:
            self.bias = torch.nn.Parameter(
                torch.zeros(bias_shape).uniform_(-bias_bound, bias_bound), requires_grad=True
            )
        else:
            self.bias = None

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: str | None,
        pre_reshape_lengths: dict[str, int] | None,
        post_reshape_pattern: str | None,
        post_reshape_lengths: dict[str, int] | None,
    ) -> None:
        self.pre_rearrange: Rearrange | None = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict[str, int], pre_reshape_lengths))

        self.post_rearrange: Rearrange | None = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict[str, int], post_reshape_lengths))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
