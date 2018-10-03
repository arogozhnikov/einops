import itertools
from collections import OrderedDict
from typing import Tuple, List, Set

import numpy

from backends import get_backend

CompositeAxis = List[str]
_reductions = ('min', 'max', 'sum', 'mean', 'prod')


def reduce_axes(tensor, reduction_type, reduced_axes: Tuple[int]):
    reduced_axes = tuple(reduced_axes)
    if len(reduced_axes) == 0:
        return tensor
    assert reduction_type in _reductions
    if reduction_type == 'mean':
        if not get_backend(tensor).is_float_type(tensor):
            raise NotImplementedError('reduce_mean is not available for non-floating tensors')
    return get_backend(tensor).reduce(tensor, reduction_type, reduced_axes)


class TransformRecipe:
    def __init__(self,
                 elementary_axes_lengths: List,
                 # list of expressions (or just sizes) for elementary axes as they appear in left expression
                 input_composite_axes: List[Tuple[List[int], List[int]]],
                 # each dimension in input can help to reconstruct or verify one of dimensions
                 output_composite_axes: List[List[int]],  # ids of axes as they appear in result
                 reduction_type: str = 'none',
                 reduced_elementary_axes: Tuple[int] = (),
                 ellipsis_positions: Tuple[int, int] = (1000, 1000),
                 ):
        self.axes_lengths = elementary_axes_lengths
        self.input_composite_axes = input_composite_axes
        self.output_composite_axes = output_composite_axes
        self.final_axes_grouping_flat = list(itertools.chain(*output_composite_axes))
        self.reduction_type = reduction_type
        self.reduced_elementary_axes = reduced_elementary_axes
        self.ellipsis_positions = ellipsis_positions

    def reconstruct_from_shape(self, shape):
        axes_lengths = list(self.axes_lengths)
        if self.ellipsis_positions != (1000, 1000):
            assert len(shape) >= len(self.input_composite_axes) - 1
        else:
            assert len(shape) == len(self.input_composite_axes)
        for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
            before_ellipsis = input_axis
            after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
            if input_axis == self.ellipsis_positions[0]:
                assert len(known_axes) == 0
                unknown_axis, = unknown_axes
                ellipsis_shape = shape[before_ellipsis:after_ellipsis + 1]
                axes_lengths[unknown_axis] = numpy.prod(ellipsis_shape, dtype=int)
            else:
                if input_axis < self.ellipsis_positions[0]:
                    length = shape[before_ellipsis]
                else:
                    length = shape[after_ellipsis]
                known_product = 1
                for axis in known_axes:
                    known_product *= axes_lengths[axis]

                if len(unknown_axes) == 0:
                    if isinstance(length, int) and isinstance(known_product, int):
                        assert length == known_product
                else:
                    if isinstance(length, int) and isinstance(known_product, int):
                        assert length % known_product == 0
                    unknown_axis, = unknown_axes
                    axes_lengths[unknown_axis] = length // known_product

        init_shapes = axes_lengths
        reduced_axes_lengths = [dim for i, dim in enumerate(axes_lengths) if i not in self.reduced_elementary_axes]
        final_shapes = []
        for output_axis, grouping in enumerate(self.output_composite_axes):
            if output_axis == self.ellipsis_positions[1]:
                final_shapes.extend(ellipsis_shape)
            else:
                group_length = 1
                for elementary_axis in grouping:
                    group_length = group_length * reduced_axes_lengths[elementary_axis]
                final_shapes.append(group_length)
        return init_shapes, final_shapes

    def reduce(self, tensor):
        return reduce_axes(tensor, reduction_type=self.reduction_type, reduced_axes=self.reduced_elementary_axes)

    def apply(self, tensor):
        backend = get_backend(tensor)
        init_shapes, final_shapes = self.reconstruct_from_shape(backend.shape(tensor))
        tensor = self.reduce(backend.reshape(tensor, init_shapes))
        tensor = backend.transpose(tensor, self.final_axes_grouping_flat)
        return backend.reshape(tensor, final_shapes)


def parse_expression(expression: str) -> Tuple[Set[str], List[CompositeAxis]]:
    """
    Parses an indexing expression (for a single tensor).
    Checks uniqueness of names, checks usage of '...'
    Returns set of all used identifiers and a list of axis groups
    """
    identifiers = set()
    composite_axes = []
    if '.' in expression:
        assert ('...' in expression) and (str.count(expression, '...') == 1) and (str.count(expression, '.') == 3)
        expression = expression.replace('...', '.')

    bracket_group = None

    def add_axis_name(x):
        if x is not None:
            if x in identifiers:
                raise ValueError('Indexing expression contains duplicate dimension "{x}"')
            identifiers.add(x)
            if bracket_group is None:
                composite_axes.append([x])
            else:
                bracket_group.append(x)

    current_identifier = None
    for char in expression:
        if char in '(). ':
            add_axis_name(current_identifier)
            current_identifier = None
            if char == '.':
                assert bracket_group is None
                # TODO replace with unicode ellipsis
                composite_axes.append('.')
                identifiers.add('.')
            elif char == '(':
                assert bracket_group is None
                bracket_group = []
            elif char == ')':
                assert bracket_group is not None
                composite_axes.append(bracket_group)
                bracket_group = None
        elif '0' <= char <= '9':
            assert current_identifier is not None
            current_identifier += char
        elif 'a' <= char <= 'z':
            if current_identifier is None:
                current_identifier = char
            else:
                current_identifier += char
        else:
            raise RuntimeError("Unknown character '{}'".format(char))

    if bracket_group is not None:
        raise ValueError('Imbalanced parentheses in expression: "{}"'.format(expression))
    add_axis_name(current_identifier)
    return identifiers, composite_axes


def _parse_composite_axis(composite_axis_name: str):
    axes_names = [axis for axis in composite_axis_name.split(' ') if len(axis) > 0]
    for axis in axes_names:
        if axis == '_':
            continue
        assert 'a' <= axis[0] <= 'z'
        for letter in axis:
            assert str.isdigit(letter) or 'a' <= letter <= 'z'
    return axes_names


def _check_elementary_axis_name(name: str) -> bool:
    """
    Valid elementary axes contain only lower latin letters and digits and start with a letter.
    """
    if len(name) == 0:
        return False
    if not 'a' <= name[0] <= 'z':
        return False
    for letter in name:
        if (not letter.isdigit()) and not ('a' <= letter <= 'z'):
            return False
    return True


# TODO parenthesis within brackets
# TODO add logaddexp
def reduce(tensor, pattern, operation, **axes_lengths):
    assert operation in ['none', 'min', 'max', 'sum', 'mean', 'prod']
    left, right = pattern.split('->')
    # checking that both have similar letters
    identifiers_left, composite_axes_left = parse_expression(left)
    identifiers_rght, composite_axes_rght = parse_expression(right)

    if operation == 'none':
        difference = set.symmetric_difference(identifiers_left, identifiers_rght)
        if len(difference) > 0:
            raise RuntimeError('Identifiers were only one side of expression: {}'.format(difference))
    else:
        difference = set.difference(identifiers_rght, identifiers_left)
        if len(difference) > 0:
            raise RuntimeError('Unexpected identifiers appeared on the right side of expression: {}'.format(difference))

    # parsing all dimensions to find out lengths
    known_lengths = OrderedDict()
    position_lookup = {}
    position_lookup_after_reduction = {}
    reduced_axes = []
    for composite_axis in composite_axes_left:
        for axis in composite_axis:
            position_lookup[axis] = len(position_lookup)
            if axis in identifiers_rght:
                position_lookup_after_reduction[axis] = len(position_lookup_after_reduction)
            else:
                reduced_axes.append(len(known_lengths))
            known_lengths[axis] = None

    def update_axis_length(axis_name, axis_length):
        if known_lengths[axis_name] is not None:
            # TODO add check for static graphs?
            if isinstance(axis_length, int) and isinstance(known_lengths[axis_name], int):
                if axis_length != known_lengths[axis_name]:
                    raise RuntimeError('Value for {} was inferred to be {} not {}'.format(
                        axis_name, axis_length, known_lengths[axis_name]))
        else:
            known_lengths[axis_name] = axis_length

    for elementary_axis, axis_length in axes_lengths.items():
        if not _check_elementary_axis_name(elementary_axis):
            raise RuntimeError('Invalid name for an axis', elementary_axis)
        update_axis_length(elementary_axis, axis_length)

    input_axes_known_unknown = []
    # inferring rest of sizes from arguments
    for composite_axis in composite_axes_left:
        known = {axis for axis in composite_axis if known_lengths[axis] is not None}
        unknown = {axis for axis in composite_axis if known_lengths[axis] is None}
        lookup = dict(zip(list(known_lengths), range(len(known_lengths))))
        if len(unknown) > 1:
            raise RuntimeError('', )
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(([lookup[axis] for axis in known], [lookup[axis] for axis in unknown]))

    result_axes_grouping = [[position_lookup_after_reduction[axis] for axis in composite_axis]
                            for composite_axis in composite_axes_rght]

    ellipsis_left = 1000 if '.' not in composite_axes_left else composite_axes_left.index('.')
    ellipsis_rght = 1000 if '.' not in composite_axes_rght else composite_axes_rght.index('.')

    recipe = TransformRecipe(elementary_axes_lengths=list(known_lengths.values()),
                             input_composite_axes=input_axes_known_unknown,
                             output_composite_axes=result_axes_grouping,
                             reduction_type=operation,
                             reduced_elementary_axes=tuple(reduced_axes),
                             ellipsis_positions=(ellipsis_left, ellipsis_rght)
                             )

    return recipe.apply(tensor=tensor)


def transpose(tensor, pattern, **axes_lengths):
    """
    einops.transpose is a reader-friendly smart element reordering for multidimensional tensors.
    This operation replaces usual transpose (axes permutation), reshape (view), squeeze, unsqueeze, and
    other operations.

    :param tensor: tensor of any supported library (e.g. numpy.ndarray).
            list of tensors is also accepted, those should be of the same type and shape
    :param pattern: string, transposition pattern
    :param axes_lengths: any additional specifications fpr dimensions
    :return: tensor of the same type as input. If possible, a view to the original tensor is returned.

    Examples:
    >>> # suppose we have a set of images in "h w c" format (height-width-channel)
    >>> images = [numpy.random.randn(30, 40, 3) for _ in range(32)]
    >>> transpose(images, 'b h w c -> b h w c').shape # stacked along first (batch) axis
    (32, 30, 40, 3)
    >>> transpose(images, 'b h w c -> (b h) w c').shape # concatenated images along height (vertical axis)
    (960, 40, 3)      # 960 = 32 * 30
    >>> transpose(images, 'b h w c -> h (b w) c').shape # concatenated images along horizontal axis
    (30, 1280, 3)     # 1280 = 32 * 40
    >>> transpose(images, 'b h w c -> b c h w').shape # reordered axes to "b c h w" format for deep learning
    (32, 3, 30, 40)
    >>> transpose(images, 'b h w c -> b (c h w)').shape # flattened each image into a vector
    (32, 3600)        # 3600 = 30 * 40 * 3
    >>> transpose(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape # split each image into 4 smaller
    (128, 15, 20, 3)  # 128 = 32 * 2 * 2

    When composing axes, C-order enumeration used (consecutive elements have different last axis)
    More examples and explanations can be found in the einops guide.
    """
    if isinstance(tensor, list):
        if len(tensor) == 0:
            raise TypeError("Transposition can't be applied to an empty list")
        tensor = get_backend(tensor[0]).stack_on_zeroth_dimension(tensor)
    return reduce(tensor, pattern, operation='none', **axes_lengths)


def check_shapes(*shapes: List[dict], **lengths):
    for shape in shapes:
        assert isinstance(shape, dict)
        for axis_name, axis_length in shape.items():
            assert isinstance(axis_length, int)
            if axis_name in lengths:
                # TODO static frameworks?
                assert lengths[axis_name] == axis_length
            else:
                lengths[axis_name] = axis_length


def parse_shape(x, names: str):
    """
    Parse a tensor shape to dictionary mapping axes names to their lengths.
    Underscores are for

    >>> x = numpy.zeros([2, 3, 5, 7])
    >>> parse_shape(x, 'batch _ h w')
    {'batch': 2, 'h': 5, 'w': 7}

    """
    names = [elementary_axis for elementary_axis in names.split(' ') if len(elementary_axis) > 0]
    shape = get_backend(x).shape(x)
    assert len(shape) == len(names)
    result = {}
    for axis_name, axis_length in zip(names, shape):
        if axis_name != '_':
            result[axis_name] = axis_length
    return result


def _enumerate_directions(x):
    """
    For an n-dimensional tensor, returns tensors to enumerate each axis.
    >>> i, j, k = _enumerate_directions(numpy.zeros([2, 3, 4]))
    >>> result = i + 2 * j + 3 * k

    result[i, j, k] = i + 2 * j + 3 * k, and also ot has the same shape
    """
    backend = get_backend(x)
    shape = backend.shape(x)
    result = []
    for axis_id, axis_length in enumerate(shape):
        shape = [1] * len(shape)
        shape[axis_id] = axis_length
        result.append(backend.reshape(backend.arange(0, axis_length), shape))
    return result


def asnumpy(tensor):
    """
    Convert tensor of imperative frameworks (numpy/cupy/torch/gluon/etc.) to numpy
    """
    return get_backend(tensor).to_numpy(tensor)
