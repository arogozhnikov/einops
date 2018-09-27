import itertools
from typing import Tuple, List, Set
from collections import OrderedDict

import numpy
# TODO get rid of these mandatory imports
import torch
import mxnet
import tensorflow as tf
import cupy
import chainer

CompositeAxis = List[str]


def reduce_axes(tensor, reduction_type, reduced_axes):
    reduced_axes = tuple(reduced_axes)
    if reduction_type == 'none' or len(reduced_axes) == 0:
        return tensor
    assert reduction_type in ['min', 'max', 'sum', 'mean', 'logaddexp', 'prod']
    if reduction_type == 'mean':
        # TODO check that dtype is float or double.
        raise NotImplementedError()
    if isinstance(tensor, (numpy.ndarray, mxnet.ndarray.ndarray.NDArray, cupy.ndarray)):
        return getattr(tensor, reduction_type)(axis=reduced_axes)
    elif isinstance(tensor, tf.Tensor):
        return getattr(tf, 'reduce_' + reduction_type)(tensor, axis=reduced_axes)
    elif isinstance(tensor, chainer.Variable):
        return getattr(chainer.functions, reduction_type)(tensor, axis=reduced_axes)
    elif isinstance(tensor, torch.Tensor):
        for axis in sorted(reduced_axes, reverse=True):
            if reduction_type == 'min':
                tensor, _ = tensor.min(dim=axis)
            elif reduction_type == 'max':
                tensor, _ = tensor.max(dim=axis)
            elif reduction_type == 'sum':
                tensor = tensor.sum(dim=axis)
            else:
                raise NotImplementedError()
        return tensor
    else:
        raise NotImplementedError()


class TransformRecipe:
    def __init__(self,
                 elementary_axes_lengths: List,
                 # list of expressions (or just sizes) for elementary axes as they appear in left expression
                 assignment_sequence: List[Tuple[int, List[int]]],
                 # each dimension in input can help to reconstruct or verify one of dimensions
                 result_axes_grouping: List[List[int]],  # ids of axes as they appear in result
                 reduction_type: str = 'none',
                 reduced_elementary_axes: List[int] = (),
                 ):
        self.axes_lengths = elementary_axes_lengths
        self.assignment_sequence = assignment_sequence
        self.final_axes_grouping = result_axes_grouping
        self.final_axes_grouping_flat = list(itertools.chain(*result_axes_grouping))
        self.reduction_type = reduction_type
        self.reduced_elementary_axes = reduced_elementary_axes

    def reconstruct_from_shape(self, shape):
        axes_lengths = list(self.axes_lengths)
        for input_axis, (axis, denominator_axes) in enumerate(self.assignment_sequence):
            length = shape[input_axis]
            for denominator_axis in denominator_axes:
                if isinstance(length, int) and isinstance(axes_lengths[denominator_axis], int):
                    # TODO check for static-graph frameworks
                    assert length % axes_lengths[denominator_axis] == 0
                length = length // axes_lengths[denominator_axis]
            if axes_lengths[axis] is not None:
                # checking dimension
                if isinstance(axes_lengths[axis], int) and isinstance(length, int):
                    print('checked dimension')
                    # TODO check for static graphs
                    assert axes_lengths[axis] == length
            else:
                axes_lengths[axis] = length

        init_shapes = axes_lengths
        reduced_axes_lengths = [dim for i, dim in enumerate(axes_lengths) if i not in self.reduced_elementary_axes]
        final_shapes = []
        for grouping in self.final_axes_grouping:
            group_length = 1
            for elementary_axis in grouping:
                group_length = group_length * reduced_axes_lengths[elementary_axis]
            final_shapes.append(group_length)
        return init_shapes, final_shapes

    def reduce(self, tensor):
        return reduce_axes(tensor, reduction_type=self.reduction_type, reduced_axes=self.reduced_elementary_axes)

    def apply(self, tensor):
        if isinstance(tensor, (numpy.ndarray, mxnet.ndarray.ndarray.NDArray, cupy.ndarray, chainer.Variable)):
            init_shapes, final_shapes = self.reconstruct_from_shape(tensor.shape)
            return self.reduce(tensor.reshape(init_shapes)) \
                .transpose(self.final_axes_grouping_flat).reshape(final_shapes)
        elif isinstance(tensor, torch.Tensor):
            init_shapes, final_shapes = self.reconstruct_from_shape(tensor.shape)
            return self.reduce(tensor.reshape(init_shapes)) \
                .permute(self.final_axes_grouping_flat).reshape(final_shapes)
        elif isinstance(tensor, (tf.Tensor, tf.Variable)):
            init_shapes, final_shapes = self.reconstruct_from_shape(tf.shape(tensor))
            tensor = self.reduce(tf.reshape(tensor, init_shapes))
            tensor = tf.transpose(tensor, self.final_axes_grouping_flat)
            return tf.reshape(tensor, final_shapes)
        else:
            raise NotImplementedError('Type of tensor was not recognized')


def parse_expression(expression: str) -> Tuple[Set[str], List[CompositeAxis]]:
    """
    Parses an indexing expression (for a single tensor).
    Checks uniqueness of names, checks usage of '...'
    Returns set of all used identifiers and a list of axis groups
    """
    identifiers = set()
    composite_axes = []
    if '.' in expression:
        assert ('...' in expression) and (str.count(expression, '...') == 2) and (str.count(expression, '.') == 6)
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
                raise NotImplementedError()
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


def get_axes_names(composite_axis_name: str):
    axes_names = [axis for axis in composite_axis_name.split(' ') if len(axis) > 0]
    for axis in axes_names:
        if axis == '_':
            continue
        assert 'a' <= axis[0] <= 'z'
        for letter in axis:
            assert str.isdigit(letter) or 'a' <= letter <= 'z'
    return axes_names


def transpose(tensor, pattern, **axes_lengths):
    left, right = pattern.split('->')
    # checking that both have similar letters
    identifiers_left, composite_axes_left = parse_expression(left)
    identifiers_rght, composite_axes_rght = parse_expression(right)

    # TODO add and delete dummy axes, add dots
    difference = set.symmetric_difference(identifiers_left, identifiers_rght)
    if len(difference) > 0:
        raise RuntimeError('Identifiers were only one side of expression: {}'.format(difference))

    # parsing all dimensions to find out lengths
    known_lengths = OrderedDict()
    for composite_axis in composite_axes_left:
        for axis in composite_axis:
            known_lengths[axis] = None

    def update_axis_length(axis_name, axis_length):
        if known_lengths[axis_name] is not None:
            # TODO add check for static graphs?
            if isinstance(axis_length, int) and isinstance(known_lengths[axis_name], int):
                assert axis_length == known_lengths[axis_name]
        else:
            known_lengths[axis_name] = axis_length

    for axis, axis_length in axes_lengths.items():
        elementary_axes = get_axes_names(axis)
        # TODO axis length is expression
        assert len(elementary_axes) == 1
        update_axis_length(elementary_axes[0], axis_length)
        # else:
        #     assert len(elementary_axes) == len(axis_length), [elementary_axes, axis_length]
        #     for c, v in zip(elementary_axes, axis_length):
        #         if c != '_':
        #             update_axis_length(c, v)

    denominator_indices = []
    # inferring rest of sizes from arguments
    for composite_axis in composite_axes_left:
        found = {axis for axis in composite_axis if known_lengths[axis] is not None}
        not_found = {axis for axis in composite_axis if known_lengths[axis] is None}
        lookup = dict(zip(list(known_lengths), range(len(known_lengths))))
        if len(not_found) == 0:
            # imitating that size of the first one was not computed
            not_found_axis = composite_axis[0]
            found.remove(not_found_axis)
            not_found.add(not_found_axis)

        assert len(not_found) == 1
        assert len(not_found) + len(found) == len(composite_axis)
        axis, = not_found
        computed_id = lookup[axis]
        denominator_ids = [lookup[axis] for axis in found]
        denominator_indices.append((computed_id, denominator_ids))

    result_axes_grouping = [[lookup[axis] for axis in composite_axis] for composite_axis in composite_axes_rght]

    recipe = TransformRecipe(elementary_axes_lengths=list(known_lengths.values()),
                             assignment_sequence=denominator_indices,
                             result_axes_grouping=result_axes_grouping,
                             reduction_type='none',
                             reduced_elementary_axes=[])

    return recipe.apply(tensor=tensor)


def reduce(tensor, pattern, operation, **axes_lengths):
    left, right = pattern.split('->')
    # checking that both have similar letters
    identifiers_left, composite_axes_left = parse_expression(left)
    identifiers_rght, composite_axes_rght = parse_expression(right)

    # TODO add and delete dummy axes, add dots
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
                assert axis_length == known_lengths[axis_name]
        else:
            known_lengths[axis_name] = axis_length

    for axis, axis_length in axes_lengths.items():
        elementary_axes = get_axes_names(axis)
        if len(elementary_axes) == 1 and isinstance(axis_length, int):
            update_axis_length(elementary_axes[0], axis_length)
        else:
            assert len(elementary_axes) == len(axis_length), [elementary_axes, axis_length]
            for c, v in zip(elementary_axes, axis_length):
                if c != '_':
                    update_axis_length(c, v)

    denominator_indices = []
    position_lookup = dict(zip(list(known_lengths), range(len(known_lengths))))
    # inferring rest of sizes from arguments
    for composite_axis in composite_axes_left:
        found = {axis for axis in composite_axis if known_lengths[axis] is not None}
        not_found = {axis for axis in composite_axis if known_lengths[axis] is None}
        if len(not_found) == 0:
            # imitating that size of the first one was not computed
            not_found_axis = composite_axis[0]
            found.remove(not_found_axis)
            not_found.add(not_found_axis)

        assert len(not_found) == 1
        assert len(not_found) + len(found) == len(composite_axis)
        axis, = not_found
        computed_id = position_lookup[axis]
        denominator_ids = [position_lookup[axis] for axis in found]
        denominator_indices.append((computed_id, denominator_ids))

    result_axes_grouping = [[position_lookup_after_reduction[axis] for axis in composite_axis]
                            for composite_axis in composite_axes_rght]

    recipe = TransformRecipe(elementary_axes_lengths=list(known_lengths.values()),
                             assignment_sequence=denominator_indices,
                             result_axes_grouping=result_axes_grouping,
                             reduction_type=operation,
                             reduced_elementary_axes=reduced_axes
                             )

    return recipe.apply(tensor=tensor)


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
    names = [elementary_axis for elementary_axis in names.split(' ') if len(elementary_axis) > 0]
    assert len(x.shape) == len(names)
    result = {}
    # TODO framework resolution?
    for axis_name, axis_length in zip(names, x.shape):
        if axis_name != '_':
            result[axis_name] = axis_length
    return result
