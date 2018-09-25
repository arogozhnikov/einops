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


class TransposeRecipe:
    def __init__(self,
                 elementary_axes_lengths: List,
                 # list of expressions (or just sizes) for elementary axes as they appear in left expression
                 assignment_sequence: List[Tuple[int, List[int]]],
                 # each dimension in input can help to reconstruct one dimension
                 result_axes_grouping: List[List[int]],  # ids of axes as they appear in result
                 ):
        self.axes_lengths = elementary_axes_lengths
        self.assignment_sequence = assignment_sequence
        self.final_axes_grouping = result_axes_grouping
        self.final_axes_grouping_flat = list(itertools.chain(*result_axes_grouping))

    def reconstruct_from_shape(self, shape):
        axes_lengths = list(self.axes_lengths)
        for input_axis, (axis, denominator_axes) in enumerate(self.assignment_sequence):
            length = shape[input_axis]
            for denominator_axis in denominator_axes:
                # TODO check that divisible, this may be impossible for static-graph frameworks
                length = length // axes_lengths[denominator_axis]
            if axes_lengths[axis] is not None:
                # checking dimension
                if isinstance(axes_lengths[axis], int) and isinstance(length, int):
                    print('checked dimension')
                    assert axes_lengths[axis] == length
                # TODO check for static graph computations
            else:
                axes_lengths[axis] = length

        init_shapes = axes_lengths
        final_shapes = []
        for grouping in self.final_axes_grouping:
            group_length = 1
            for elementary_axis in grouping:
                group_length = group_length * axes_lengths[elementary_axis]
            final_shapes.append(group_length)
        return init_shapes, final_shapes

    def apply(self, tensor):
        if isinstance(tensor, (numpy.ndarray, mxnet.ndarray.ndarray.NDArray, cupy.ndarray, chainer.Variable)):
            init_shapes, final_shapes = self.reconstruct_from_shape(tensor.shape)
            return tensor.reshape(init_shapes).transpose(self.final_axes_grouping_flat).reshape(final_shapes)
        elif isinstance(tensor, torch.Tensor):
            init_shapes, final_shapes = self.reconstruct_from_shape(tensor.shape)
            return tensor.reshape(*init_shapes).permute(self.final_axes_grouping_flat).reshape(final_shapes)
        elif isinstance(tensor, tf.Tensor):
            init_shapes, final_shapes = self.reconstruct_from_shape(tf.shape(tensor))
            tensor = tf.reshape(tensor, init_shapes)
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
        assert ('...' in expression) and (str.count(expression, '.') == 3)
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
        if char in '()., ':
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
            add_axis_name(current_identifier + char)
            current_identifier = None
        elif 'a' <= char <= 'z':
            add_axis_name(current_identifier)
            current_identifier = char
        else:
            raise RuntimeError("Unknown character '{}'".format(char))

    if bracket_group is not None:
        raise ValueError('Imbalanced parentheses in expression: "{}"'.format(expression))
    add_axis_name(current_identifier)
    return identifiers, composite_axes


def get_axes_names(composite_axis_name: str):
    # TODO add spaces and long names
    elementary_axes = []
    composite_axis_name = list(composite_axis_name)
    while len(composite_axis_name) > 0:
        letter = composite_axis_name.pop()
        if 'a' <= letter <= 'z' or letter == '_':
            elementary_axes.append(letter)
        else:
            assert '0' <= letter <= '9'
            prev_letter = composite_axis_name.pop()
            assert 'a' <= prev_letter <= 'z'
            elementary_axes.append(prev_letter + letter)
    return elementary_axes[::-1]


def transpose(tensor, pattern, **axes_lengths):
    left, right = pattern.split('->')
    # checking that both have similar letters
    identifiers_left, composite_axes_left = parse_expression(left)
    identifiers_rght, composite_axes_rght = parse_expression(right)

    # TODO add and delete dummy axes, add dots
    difference = set.difference(identifiers_left, identifiers_rght)
    if len(difference) > 0:
        raise RuntimeError('Identifiers were only one side of expression: {}'.format(difference))

    # parsing all dimensions to find out lengths
    known_lengths = OrderedDict()
    for composite_axis in composite_axes_left:
        for axis in composite_axis:
            known_lengths[axis] = None

    def update_axis_length(axis_name, axis_length):
        # assert axis_length > 0
        # axis_length = int(axis_length)
        if known_lengths[axis_name] is not None:
            # this one may require special calculation
            assert axis_length == known_lengths[axis_name]
        else:
            known_lengths[axis_name] = axis_length

    for axis, axis_length in axes_lengths.items():
        elementary_axes = get_axes_names(axis)
        if len(elementary_axes) == 1 and isinstance(axis_length, int):
            update_axis_length(elementary_axes[0], axis_length)
        else:
            assert len(elementary_axes) == len(axis_length), [elementary_axes, axis_length]
            for c, v in zip(axis, axis_length):
                if c != '_':
                    update_axis_length(c, v)

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

    recipe = TransposeRecipe(elementary_axes_lengths=list(known_lengths.values()),
                             assignment_sequence=denominator_indices,
                             result_axes_grouping=result_axes_grouping)

    return recipe.apply(tensor=tensor)
