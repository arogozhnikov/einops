from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, cast

from .einops import EinopsError, Reduction, _apply_recipe_array_api, _prepare_transformation_recipe
from .packing import _unpack_split_positions, analyze_pattern

if TYPE_CHECKING:
    # avoid runtime dependencies
    import numpy as np
    from typing_extensions import CapsuleType

    class ArrayAPITensor(Protocol):
        def __array_namespace__(self, /) -> ModuleType: ...
        @property
        def shape(self, /) -> tuple[int, ...]: ...
        @property
        def ndim(self, /) -> int: ...
        def __getitem__(self, arg) -> "ArrayAPITensor": ...
        # additional
        def __dlpack__(self, /, *, stream: None = ...) -> CapsuleType: ...
else:

    class ArrayAPITensor(Protocol):
        def __array_namespace__(self, /) -> ModuleType: ...
        @property
        def shape(self, /) -> tuple[int, ...]: ...
        @property
        def ndim(self, /) -> int: ...
        def __getitem__(self, arg) -> "ArrayAPITensor": ...


Tensor = TypeVar("Tensor", bound=ArrayAPITensor)


def reduce(tensor: Tensor | list[Tensor], pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor:
    """Same as einops.repeat, restricted to array-api-compliant tensors. See einops.repeat for details.
    Supported reductions: "min", "max", "sum", "mean", "prod", "any", "all"
    """
    if isinstance(tensor, list):
        if len(tensor) == 0:
            raise TypeError("Einops can't be applied to an empty list")
        xp = tensor[0].__array_namespace__()
        tensor = cast(Tensor, xp.stack(tensor))
    else:
        xp = tensor.__array_namespace__()
    try:
        hashable_axes_lengths = tuple(axes_lengths.items())
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_names=tuple(axes_lengths), ndim=tensor.ndim)
        return _apply_recipe_array_api(
            xp,
            recipe=recipe,
            tensor=tensor,
            reduction_type=reduction,
            axes_lengths=hashable_axes_lengths,
        )
    except EinopsError as e:
        message = f' Error while processing {reduction}-reduction pattern "{pattern}".'
        if not isinstance(tensor, list):
            message += f"\n Input tensor shape: {tensor.shape}. "
        else:
            message += "\n Input is list. "
        message += f"Additional info: {axes_lengths}."
        raise EinopsError(message + f"\n {e}") from None


def repeat(tensor: Tensor | list[Tensor], pattern: str, **axes_lengths: int) -> Tensor:
    """Same as einops.repeat, restricted to array-api-compliant tensors. See einops.repeat for details."""
    return reduce(tensor, pattern, reduction="repeat", **axes_lengths)


def rearrange(tensor: Tensor | list[Tensor], pattern: str, **axes_lengths: int) -> Tensor:
    """Same as einops.rearrange, restricted to array-api-compliant tensors. See einops.rearrange for details."""
    return reduce(tensor, pattern, reduction="rearrange", **axes_lengths)


def asnumpy(tensor: Tensor) -> "np.ndarray":
    import numpy as np

    return np.from_dlpack(tensor)


Shape: TypeAlias = tuple


def pack(tensors: Sequence[Tensor], pattern: str) -> tuple[Tensor, list[Shape]]:
    """Same as einops.pack, restricted to array-api-compliant tensors. See einops.pack for details."""
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, "pack")
    xp = tensors[0].__array_namespace__()

    reshaped_tensors: list[Tensor] = []
    packed_shapes: list[Shape] = []
    for i, tensor in enumerate(tensors):
        shape = tensor.shape
        if len(shape) < min_axes:
            raise EinopsError(
                f"packed tensor #{i} (enumeration starts with 0) has shape {shape}, "
                f"while pattern {pattern} assumes at least {min_axes} axes"
            )
        axis_after_packed_axes = len(shape) - n_axes_after
        packed_shapes.append(shape[n_axes_before:axis_after_packed_axes])
        reshaped_tensors.append(xp.reshape(tensor, (*shape[:n_axes_before], -1, *shape[axis_after_packed_axes:])))

    return xp.concat(reshaped_tensors, axis=n_axes_before), packed_shapes


def unpack(tensor: Tensor, packed_shapes: Sequence[Shape | list], pattern: str) -> list[Tensor]:
    """Same as einops.unpack, restricted to array-api-compliant tensors. See einops.unpack for details."""
    xp = tensor.__array_namespace__()
    n_axes_before, n_axes_after, min_axes = analyze_pattern(pattern, opname="unpack")

    # backend = get_backend(tensor)
    input_shape = tensor.shape
    if len(input_shape) != n_axes_before + 1 + n_axes_after:
        raise EinopsError(f"unpack(..., {pattern}) received input of wrong dim with shape {input_shape}")

    unpacked_axis: int = n_axes_before

    split_positions = _unpack_split_positions(packed_shapes, input_shape[unpacked_axis], pattern)

    shape_start = input_shape[:unpacked_axis]
    shape_end = input_shape[unpacked_axis + 1 :]
    slice_filler = (slice(None, None),) * unpacked_axis
    try:
        return [
            xp.reshape(
                # shortest way slice arbitrary axis
                tensor[(*slice_filler, slice(split_positions[i], split_positions[i + 1]), ...)],
                (*shape_start, *element_shape, *shape_end),
            )
            for i, element_shape in enumerate(packed_shapes)
        ]
    except Exception as e:
        # this hits if there is an error during reshapes, which means passed shapes were incorrect
        raise EinopsError(
            f'Error during unpack(..., "{pattern}"): could not split axis of size {split_positions[-1]}'
            f" into requested {packed_shapes}"
        ) from e
