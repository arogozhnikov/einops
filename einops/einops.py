import functools
import itertools
import string
import typing
from collections import OrderedDict
from typing import Set, Tuple, List, Dict, Union, Callable, Optional, TypeVar, cast, Any

if typing.TYPE_CHECKING:
    import numpy as np

from . import EinopsError
from ._backends import get_backend
from .parsing import ParsedExpression, _ellipsis, AnonymousAxis

Tensor = TypeVar('Tensor')
ReductionCallable = Callable[[Tensor, Tuple[int, ...]], Tensor]
Reduction = Union[str, ReductionCallable]

_reductions = ('min', 'max', 'sum', 'mean', 'prod')
# magic integers are required to stay within
# traceable subset of language
_ellipsis_not_in_parenthesis: List[int] = [-999]
_unknown_axis_length = -999999


def is_ellipsis_not_in_parenthesis(group: List[int]) -> bool:
    if len(group) != 1:
        return False
    return group[0] == -999


def _product(sequence: List[int]) -> int:
    """ minimalistic product that works both with numbers and symbols. Supports empty lists """
    result = 1
    for element in sequence:
        result *= element
    return result


def _reduce_axes(tensor, reduction_type: Reduction, reduced_axes: List[int], backend):
    if callable(reduction_type):
        # custom callable
        return reduction_type(tensor, tuple(reduced_axes))
    else:
        # one of built-in operations
        if len(reduced_axes) == 0:
            return tensor
        assert reduction_type in _reductions
        if reduction_type == 'mean':
            if not backend.is_float_type(tensor):
                raise NotImplementedError('reduce_mean is not available for non-floating tensors')
        return backend.reduce(tensor, reduction_type, tuple(reduced_axes))


def _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes):
    # 'collapses' neighboring axes if those participate in the result pattern in the same order
    # TODO add support for added_axes
    assert len(axes_reordering) + len(reduced_axes) == len(init_shapes)
    # joining consecutive axes that will be reduced
    # possibly we can skip this if all backends can optimize this (not sure)
    reduced_axes = tuple(sorted(reduced_axes))
    for i in range(len(reduced_axes) - 1)[::-1]:
        if reduced_axes[i] + 1 == reduced_axes[i + 1]:
            removed_axis = reduced_axes[i + 1]
            removed_length = init_shapes[removed_axis]
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            reduced_axes = reduced_axes[:i + 1] + tuple(axis - 1 for axis in reduced_axes[i + 2:])

    # removing axes that are moved together during reshape
    def build_mapping():
        init_to_final = {}
        for axis in range(len(init_shapes)):
            if axis in reduced_axes:
                init_to_final[axis] = None
            else:
                after_reduction = sum(x is not None for x in init_to_final.values())
                init_to_final[axis] = list(axes_reordering).index(after_reduction)
        return init_to_final

    init_axis_to_final_axis = build_mapping()

    for init_axis in range(len(init_shapes) - 1)[::-1]:
        if init_axis_to_final_axis[init_axis] is None:
            continue
        if init_axis_to_final_axis[init_axis + 1] is None:
            continue
        if init_axis_to_final_axis[init_axis] + 1 == init_axis_to_final_axis[init_axis + 1]:
            removed_axis = init_axis + 1
            removed_length = init_shapes[removed_axis]
            removed_axis_after_reduction = sum(x not in reduced_axes for x in range(removed_axis))

            reduced_axes = tuple(axis if axis < removed_axis else axis - 1 for axis in reduced_axes)
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            old_reordering = axes_reordering
            axes_reordering = []
            for axis in old_reordering:
                if axis == removed_axis_after_reduction:
                    pass
                elif axis < removed_axis_after_reduction:
                    axes_reordering.append(axis)
                else:
                    axes_reordering.append(axis - 1)
            init_axis_to_final_axis = build_mapping()

    return init_shapes, reduced_axes, axes_reordering, final_shapes


CookedRecipe = Tuple[List[int], List[int], List[int], Dict[int, int], List[int]]


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    # structure is non-mutable. In future, this can be non-mutable dataclass (python 3.7+)

    def __init__(self,
                 # list of expressions (or just sizes) for elementary axes as they appear in left expression.
                 # this is what (after computing unknown parts) will be a shape after first transposition.
                 # If ellipsis is present, it forms one dimension here (in the right position).
                 elementary_axes_lengths: List[int],
                 # each dimension in input can help to reconstruct length of one elementary axis
                 # or verify one of dimensions. Each element points to element of elementary_axes_lengths
                 input_composite_axes: List[Tuple[List[int], List[int]]],
                 # indices of axes to be squashed
                 reduced_elementary_axes: List[int],
                 # in which order should axes be reshuffled after reduction
                 axes_permutation: List[int],
                 # at which positions which of elementary axes should appear
                 added_axes: Dict[int, int],
                 # ids of axes as they appear in result, again pointers to elementary_axes_lengths,
                 # only used to infer result dimensions
                 output_composite_axes: List[List[int]],
                 # positions of ellipsis in lhs and rhs of expression
                 ellipsis_position_in_lhs: Optional[int] = None,
                 ):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.input_composite_axes: List[Tuple[List[int], List[int]]] = input_composite_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes
        self.axes_permutation: List[int] = axes_permutation
        self.added_axes: Dict[int, int] = added_axes
        # This is redundant information, but more convenient to use
        self.reduced_elementary_axes: List[int] = reduced_elementary_axes
        # setting to a large number to avoid handling Nones in reconstruct_from_shape
        self.ellipsis_position_in_lhs: int = ellipsis_position_in_lhs if ellipsis_position_in_lhs is not None else 10000


def _reconstruct_from_shape_uncached(self: TransformRecipe, shape: List[int]) -> CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, keras, theano) and UnknownSize (keras, mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    if self.ellipsis_position_in_lhs != 10000:
        if len(shape) < len(self.input_composite_axes) - 1:
            raise EinopsError('Expected at least {} dimensions, got {}'.format(
                len(self.input_composite_axes) - 1, len(shape)))
    else:
        if len(shape) != len(self.input_composite_axes):
            raise EinopsError('Expected {} dimensions, got {}'.format(len(self.input_composite_axes), len(shape)))

    ellipsis_shape: List[int] = []
    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
        before_ellipsis = input_axis
        after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
        if input_axis == self.ellipsis_position_in_lhs:
            assert len(known_axes) == 0 and len(unknown_axes) == 1
            unknown_axis: int = unknown_axes[0]
            ellipsis_shape = shape[before_ellipsis:after_ellipsis + 1]
            for d in ellipsis_shape:
                if d is None:
                    raise EinopsError("Couldn't infer shape for one or more axes represented by ellipsis")
            total_dim_size: int = _product(ellipsis_shape)
            axes_lengths[unknown_axis] = total_dim_size
        else:
            if input_axis < self.ellipsis_position_in_lhs:
                length = shape[before_ellipsis]
            else:
                length = shape[after_ellipsis]
            known_product = 1
            for axis in known_axes:
                known_product *= axes_lengths[axis]

            if len(unknown_axes) == 0:
                if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                    raise EinopsError('Shape mismatch, {} != {}'.format(length, known_product))
            # this is enforced when recipe is created
            # elif len(unknown_axes) > 1:
            #     raise EinopsError(
            #         "Lengths of two or more axes in parenthesis not provided (dim={}), can't infer dimensions".
            #             format(known_product)
            #     )
            else:
                if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                    raise EinopsError("Shape mismatch, can't divide axis of length {} in chunks of {}".format(
                        length, known_product))

                unknown_axis = unknown_axes[0]
                inferred_length: int = length // known_product
                axes_lengths[unknown_axis] = inferred_length

    # at this point all axes_lengths are computed (either have values or variables, but not Nones)

    # TODO more readable expression
    init_shapes = axes_lengths[:len(axes_lengths) - len(self.added_axes)]
    final_shapes: List[int] = []
    for output_axis, grouping in enumerate(self.output_composite_axes):
        if is_ellipsis_not_in_parenthesis(grouping):
            final_shapes.extend(ellipsis_shape)
        else:
            lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
            final_shapes.append(_product(lengths))
    reduced_axes = self.reduced_elementary_axes
    axes_reordering = self.axes_permutation
    added_axes: Dict[int, int] = {
        pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()}
    # if optimize:
    #     assert len(self.added_axes) == 0
    #     return _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes)
    return init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes


_reconstruct_from_shape = functools.lru_cache(1024)(_reconstruct_from_shape_uncached)


def _apply_recipe(recipe: TransformRecipe, tensor: Tensor, reduction_type: Reduction) -> Tensor:
    # this method works for all backends but not compilable with
    backend = get_backend(tensor)

    # Ideally we'd check if any of the sizes are of type `torch.SymInt`,
    # but we don't want einops to have a dependency on torch
    tracing_dynamic_shapes = any(type(x) != int for x in tensor.shape)
    # torch.SymInt is not hashable, and will only show up here if we are in the middle of tracing
    # during torch.compile, so it is ok to be a bit slower and not cache.
    reconstruct_fn = _reconstruct_from_shape_uncached if tracing_dynamic_shapes else _reconstruct_from_shape

    init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes = \
        reconstruct_fn(recipe, backend.shape(tensor))
    tensor = backend.reshape(tensor, init_shapes)
    tensor = _reduce_axes(tensor, reduction_type=reduction_type, reduced_axes=reduced_axes, backend=backend)
    tensor = backend.transpose(tensor, axes_reordering)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=len(axes_reordering) + len(added_axes), pos2len=added_axes)
    return backend.reshape(tensor, final_shapes)


@functools.lru_cache(256)
def _prepare_transformation_recipe(pattern: str,
                                   operation: Reduction,
                                   axes_lengths: Tuple[Tuple, ...]) -> TransformRecipe:
    """ Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split('->')
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)

    # checking that axes are in agreement - new axes appear only in repeat, while disappear only in reduction
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError('Ellipsis found in right side, but not left side of a pattern {}'.format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError('Ellipsis is parenthesis in the left side is not allowed: {}'.format(pattern))
    if operation == 'rearrange':
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError('Non-unitary anonymous axes are not supported in rearrange (exception is length 1)')
        if len(difference) > 0:
            raise EinopsError('Identifiers only on one side of expression (should be on both): {}'.format(difference))
    elif operation == 'repeat':
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the left side of repeat: {}'.format(difference))
        axes_without_size = set.difference({ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)},
                                           {*left.identifiers, *(ax for ax, _ in axes_lengths)})
        if len(axes_without_size) > 0:
            raise EinopsError('Specify sizes for new axes in repeat: {}'.format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the right side of reduce {}: {}'.format(operation, difference))
    else:
        raise EinopsError('Unknown reduction {}. Expect one of {}.'.format(operation, _reductions))

    # parsing all dimensions to find out lengths
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    for composite_axis in left.composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length

    # axis_ids_after_first_reshape = range(len(axis_name2known_length)) at this point

    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)

    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}
    reduced_axes: List[int] = [position for axis, position in axis_name2position.items() if
                               axis not in rght.identifiers]
    reduced_axes = list(sorted(reduced_axes))

    for elementary_axis, axis_length in axes_lengths:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError('Invalid name for an axis', elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError('Axis {} is not used in transform'.format(elementary_axis))
        axis_name2known_length[elementary_axis] = axis_length

    input_axes_known_unknown = []
    # some of shapes will be inferred later - all information is prepared for faster inference
    for composite_axis in left.composition:
        known: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError('Could not infer sizes for {}'.format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(
            ([axis_name2position[axis] for axis in known],
             [axis_name2position[axis] for axis in unknown])
        )

    axis_position_after_reduction: Dict[str, int] = {}
    for axis_name in itertools.chain(*left.composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)

    result_axes_grouping: List[List[int]] = []
    for composite_axis in rght.composition:
        if composite_axis == _ellipsis:
            result_axes_grouping.append(_ellipsis_not_in_parenthesis)
        else:
            result_axes_grouping.append([axis_name2position[axis] for axis in composite_axis])

    ordered_axis_right = list(itertools.chain(*rght.composition))
    axes_permutation = [
        axis_position_after_reduction[axis] for axis in ordered_axis_right if axis in left.identifiers]
    added_axes = {i: axis_name2position[axis_name] for i, axis_name in enumerate(ordered_axis_right)
                  if axis_name not in left.identifiers}

    ellipsis_left = None if _ellipsis not in left.composition else left.composition.index(_ellipsis)

    return TransformRecipe(
        elementary_axes_lengths=list(axis_name2known_length.values()),
        input_composite_axes=input_axes_known_unknown,
        reduced_elementary_axes=reduced_axes,
        axes_permutation=axes_permutation,
        added_axes=added_axes,
        output_composite_axes=result_axes_grouping,
        ellipsis_position_in_lhs=ellipsis_left,
    )


def reduce(tensor: Tensor, pattern: str, reduction: Reduction, **axes_lengths: int) -> Tensor:
    """
    einops.reduce provides combination of reordering and reduction using reader-friendly notation.

    Examples for reduce operation:

    ```python
    >>> x = np.random.randn(100, 32, 64)

    # perform max-reduction on the first axis
    >>> y = reduce(x, 't b c -> b c', 'max')

    # same as previous, but with clearer axes meaning
    >>> y = reduce(x, 'time batch channel -> batch channel', 'max')

    >>> x = np.random.randn(10, 20, 30, 40)

    # 2d max-pooling with kernel size = 2 * 2 for image processing
    >>> y1 = reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h2=2, w2=2)

    # if one wants to go back to the original height and width, depth-to-space trick can be applied
    >>> y2 = rearrange(y1, 'b (c h2 w2) h1 w1 -> b c (h1 h2) (w1 w2)', h2=2, w2=2)
    >>> assert parse_shape(x, 'b _ h w') == parse_shape(y2, 'b _ h w')

    # Adaptive 2d max-pooling to 3 * 4 grid
    >>> reduce(x, 'b c (h1 h2) (w1 w2) -> b c h1 w1', 'max', h1=3, w1=4).shape
    (10, 20, 3, 4)

    # Global average pooling
    >>> reduce(x, 'b c h w -> b c', 'mean').shape
    (10, 20)

    # Subtracting mean over batch for each channel
    >>> y = x - reduce(x, 'b c h w -> () c () ()', 'mean')

    # Subtracting per-image mean for each channel
    >>> y = x - reduce(x, 'b c h w -> b c () ()', 'mean')

    ```

    Parameters:
        tensor: tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, reduction pattern
        reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
            alternatively, a callable f(tensor, reduced_axes) -> tensor can be provided.
            This allows using various reductions, examples: np.max, tf.reduce_logsumexp, torch.var, etc.
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input
    """
    try:
        hashable_axes_lengths = tuple(sorted(axes_lengths.items()))
        recipe = _prepare_transformation_recipe(pattern, reduction, axes_lengths=hashable_axes_lengths)
        return _apply_recipe(recipe, tensor, reduction_type=reduction)
    except EinopsError as e:
        message = ' Error while processing {}-reduction pattern "{}".'.format(reduction, pattern)
        if not isinstance(tensor, list):
            message += '\n Input tensor shape: {}. '.format(get_backend(tensor).shape(tensor))
        else:
            message += '\n Input is list. '
        message += 'Additional info: {}.'.format(axes_lengths)
        raise EinopsError(message + '\n {}'.format(e))



def rearrange(tensor: Union[Tensor, List[Tensor]], pattern: str, **axes_lengths) -> Tensor:
    """
    einops.rearrange is a reader-friendly smart element reordering for multidimensional tensors.
    This operation includes functionality of transpose (axes permutation), reshape (view), squeeze, unsqueeze,
    stack, concatenate and other operations.

    Examples for rearrange operation:

    ```python
    # suppose we have a set of 32 images in "h w c" format (height-width-channel)
    >>> images = [np.random.randn(30, 40, 3) for _ in range(32)]

    # stack along first (batch) axis, output is a single array
    >>> rearrange(images, 'b h w c -> b h w c').shape
    (32, 30, 40, 3)

    # concatenate images along height (vertical axis), 960 = 32 * 30
    >>> rearrange(images, 'b h w c -> (b h) w c').shape
    (960, 40, 3)

    # concatenated images along horizontal axis, 1280 = 32 * 40
    >>> rearrange(images, 'b h w c -> h (b w) c').shape
    (30, 1280, 3)

    # reordered axes to "b c h w" format for deep learning
    >>> rearrange(images, 'b h w c -> b c h w').shape
    (32, 3, 30, 40)

    # flattened each image into a vector, 3600 = 30 * 40 * 3
    >>> rearrange(images, 'b h w c -> b (c h w)').shape
    (32, 3600)

    # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
    >>> rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape
    (128, 15, 20, 3)

    # space-to-depth operation
    >>> rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape
    (32, 15, 20, 12)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
                list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    if isinstance(tensor, list):
        if len(tensor) == 0:
            raise TypeError("Rearrange can't be applied to an empty list")
        tensor = get_backend(tensor[0]).stack_on_zeroth_dimension(tensor)
    return reduce(cast(Tensor, tensor), pattern, reduction='rearrange', **axes_lengths)


def repeat(tensor: Tensor, pattern: str, **axes_lengths) -> Tensor:
    """
    einops.repeat allows reordering elements and repeating them in arbitrary combinations.
    This operation includes functionality of repeat, tile, broadcast functions.

    Examples for repeat operation:

    ```python
    # a grayscale image (of shape height x width)
    >>> image = np.random.randn(30, 40)

    # change it to RGB format by repeating in each channel
    >>> repeat(image, 'h w -> h w c', c=3).shape
    (30, 40, 3)

    # repeat image 2 times along height (vertical axis)
    >>> repeat(image, 'h w -> (repeat h) w', repeat=2).shape
    (60, 40)

    # repeat image 2 time along height and 3 times along width
    >>> repeat(image, 'h w -> (h2 h) (w3 w)', h2=2, w3=3).shape
    (60, 120)

    # convert each pixel to a small square 2x2. Upsample image by 2x
    >>> repeat(image, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (60, 80)

    # pixelate image first by downsampling by 2x, then upsampling
    >>> downsampled = reduce(image, '(h h2) (w w2) -> h w', 'mean', h2=2, w2=2)
    >>> repeat(downsampled, 'h w -> (h h2) (w w2)', h2=2, w2=2).shape
    (30, 40)

    ```

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    Find more examples in einops tutorial.

    Parameters:
        tensor: tensor of any supported library (e.g. numpy.ndarray, tensorflow, pytorch).
            list of tensors is also accepted, those should be of the same type and shape
        pattern: string, rearrangement pattern
        axes_lengths: any additional specifications for dimensions

    Returns:
        Tensor of the same type as input. If possible, a view to the original tensor is returned.

    """
    return reduce(tensor, pattern, reduction='repeat', **axes_lengths)


def parse_shape(x, pattern: str) -> dict:
    """
    Parse a tensor shape to dictionary mapping axes names to their lengths.

    ```python
    # Use underscore to skip the dimension in parsing.
    >>> x = np.zeros([2, 3, 5, 7])
    >>> parse_shape(x, 'batch _ h w')
    {'batch': 2, 'h': 5, 'w': 7}

    # `parse_shape` output can be used to specify axes_lengths for other operations:
    >>> y = np.zeros([700])
    >>> rearrange(y, '(b c h w) -> b c h w', **parse_shape(x, 'b _ h w')).shape
    (2, 10, 5, 7)

    ```

    For symbolic frameworks may return symbols, not integers.

    Parameters:
        x: tensor of any of supported frameworks
        pattern: str, space separated names for axes, underscore means skip axis

    Returns:
        dict, maps axes names to their lengths
    """
    exp = ParsedExpression(pattern, allow_underscore=True)
    shape = get_backend(x).shape(x)
    if exp.has_composed_axes():
        raise RuntimeError("Can't parse shape with composite axes: {pattern} {shape}".format(
            pattern=pattern, shape=shape))
    if len(shape) != len(exp.composition):
        if exp.has_ellipsis:
            if len(shape) < len(exp.composition) - 1:
                raise RuntimeError("Can't parse shape with this number of dimensions: {pattern} {shape}".format(
                    pattern=pattern, shape=shape))
        else:
            raise RuntimeError("Can't parse shape with different number of dimensions: {pattern} {shape}".format(
                pattern=pattern, shape=shape))
    if exp.has_ellipsis:
        ellipsis_idx = exp.composition.index(_ellipsis)
        composition = (exp.composition[:ellipsis_idx] +
                       ['_'] * (len(shape) - len(exp.composition) + 1) +
                       exp.composition[ellipsis_idx + 1:])
    else:
        composition = exp.composition
    result = {}
    for (axis_name,), axis_length in zip(composition, shape):  # type: ignore
        if axis_name != '_':
            result[axis_name] = axis_length
    return result


# this one is probably not needed in the public API
def _enumerate_directions(x):
    """
    For an n-dimensional tensor, returns tensors to enumerate each axis.
    ```python
    x = np.zeros([2, 3, 4]) # or any other tensor
    i, j, k = _enumerate_directions(x)
    result = i + 2*j + 3*k
    ```

    `result[i, j, k] = i + 2j + 3k`, and also has the same shape as result
    Works very similarly to numpy.ogrid (open indexing grid)
    """
    backend = get_backend(x)
    shape = backend.shape(x)
    result = []
    for axis_id, axis_length in enumerate(shape):
        shape = [1] * len(shape)
        shape[axis_id] = axis_length
        result.append(backend.reshape(backend.arange(0, axis_length), shape))
    return result


# to avoid importing numpy
np_ndarray = Any


def asnumpy(tensor) -> np_ndarray:
    """
    Convert a tensor of an imperative framework (i.e. numpy/cupy/torch/jax/etc.) to `numpy.ndarray`

    Parameters:
        tensor: tensor of any of known imperative framework

    Returns:
        `numpy.ndarray`, converted to numpy
    """
    return get_backend(tensor).to_numpy(tensor)


def _validate_einsum_axis_name(axis_name):
    if len(axis_name) == 0:
        raise NotImplementedError("Singleton () axes are not yet supported in einsum.")
    if len(axis_name) > 1:
        raise NotImplementedError("Shape rearrangement is not yet supported in einsum.")

    axis_name = axis_name[0]

    if isinstance(axis_name, AnonymousAxis):
        raise NotImplementedError("Anonymous axes are not yet supported in einsum.")
    if len(axis_name) == 0:
        raise RuntimeError("Encountered empty axis name in einsum.")
    if not isinstance(axis_name, str):
        raise RuntimeError("Axis name in einsum must be a string.")


@functools.lru_cache(256)
def _compactify_pattern_for_einsum(pattern: str) -> str:
    if "->" not in pattern:
        # numpy allows this, so make sure users
        # don't accidentally do something like this.
        raise ValueError("Einsum pattern must contain '->'.")
    lefts_str, right_str = pattern.split('->')

    lefts = [
        ParsedExpression(left, allow_underscore=True, allow_duplicates=True)
        for left in lefts_str.split(',')
    ]

    right = ParsedExpression(right_str, allow_underscore=True)

    # Start from 'a' and go up to 'Z'
    output_axis_names = string.ascii_letters
    i = 0
    axis_name_mapping = {}

    left_patterns = []
    for left in lefts:
        left_pattern = ""
        for raw_axis_name in left.composition:

            if raw_axis_name == _ellipsis:
                left_pattern += '...'
                continue

            _validate_einsum_axis_name(raw_axis_name)
            axis_name = raw_axis_name[0]
            if axis_name not in axis_name_mapping:
                if i >= len(output_axis_names):
                    raise RuntimeError("Too many axes in einsum.")
                axis_name_mapping[axis_name] = output_axis_names[i]
                i += 1

            left_pattern += axis_name_mapping[axis_name]
        left_patterns.append(left_pattern)

    compact_pattern = ",".join(left_patterns) + "->"

    for raw_axis_name in right.composition:
        if raw_axis_name == _ellipsis:
            compact_pattern += '...'
            continue

        _validate_einsum_axis_name(raw_axis_name)
        axis_name = raw_axis_name[0]

        if axis_name not in axis_name_mapping:
            raise EinopsError(f"Unknown axis {axis_name} on right side of einsum {pattern}.")

        compact_pattern += axis_name_mapping[axis_name]

    return compact_pattern


# dunders in overloads turn arguments into positional-only.
# After python 3.7 EOL this should be replaced with '/' as the last argument.

@typing.overload
def einsum(__tensor: Tensor, __pattern: str) -> Tensor: ...
@typing.overload
def einsum(__tensor1: Tensor, __tensor2: Tensor, __pattern: str) -> Tensor: ...
@typing.overload
def einsum(__tensor1: Tensor, __tensor2: Tensor, __tensor3: Tensor, __pattern: str) -> Tensor: ...
@typing.overload
def einsum(__tensor1: Tensor, __tensor2: Tensor, __tensor3: Tensor, __tensor4: Tensor, __pattern: str) -> Tensor: ...


def einsum(*tensors_and_pattern: Union[Tensor, str]) -> Tensor:
    """
    einops.einsum calls einsum operations with einops-style named
    axes indexing, computing tensor products with an arbitrary
    number of tensors. Unlike typical einsum syntax, here you must
    pass tensors first, and then the pattern.

    Also, note that rearrange operations such as `"(batch chan) out"`,
    or singleton axes `()`, are not currently supported.

    Examples:

    For a given pattern such as:
    ```python
    >>> x, y, z = np.random.randn(3, 20, 20, 20)
    >>> output = einsum(x, y, z, "a b c, c b d, a g k -> a b k")

    ```
    the following formula is computed:
    ```tex
    output[a, b, k] =
        \sum_{c, d, g} x[a, b, c] * y[c, b, d] * z[a, g, k]
    ```
    where the summation over `c`, `d`, and `g` is performed
    because those axes names do not appear on the right-hand side.

    Let's see some additional examples:
    ```python
    # Filter a set of images:
    >>> batched_images = np.random.randn(128, 16, 16)
    >>> filters = np.random.randn(16, 16, 30)
    >>> result = einsum(batched_images, filters,
    ...                 "batch h w, h w channel -> batch channel")
    >>> result.shape
    (128, 30)

    # Matrix multiplication, with an unknown input shape:
    >>> batch_shape = (50, 30)
    >>> data = np.random.randn(*batch_shape, 20)
    >>> weights = np.random.randn(10, 20)
    >>> result = einsum(weights, data,
    ...                 "out_dim in_dim, ... in_dim -> ... out_dim")
    >>> result.shape
    (50, 30, 10)

    # Matrix trace on a single tensor:
    >>> matrix = np.random.randn(10, 10)
    >>> result = einsum(matrix, "i i ->")
    >>> result.shape
    ()

    ```

    Parameters:
        tensors_and_pattern:
            tensors: tensors of any supported library (numpy, tensorflow, pytorch, jax).
            pattern: string, einsum pattern, with commas
                separating specifications for each tensor.
                pattern should be provided after all tensors.

    Returns:
        Tensor of the same type as input, after processing with einsum.

    """
    if len(tensors_and_pattern) <= 1:
        raise ValueError(
            "`einops.einsum` takes at minimum two arguments: the tensors (at least one),"
            " followed by the pattern."
        )
    pattern = tensors_and_pattern[-1]
    if not isinstance(pattern, str):
        raise ValueError(
            "The last argument passed to `einops.einsum` must be a string,"
            " representing the einsum pattern."
        )
    tensors = tensors_and_pattern[:-1]
    pattern = _compactify_pattern_for_einsum(pattern)
    return get_backend(tensors[0]).einsum(pattern, *tensors)
