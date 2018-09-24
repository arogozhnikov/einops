import torch
import itertools
import numpy
from typing import Tuple, List, Set

CompositeAxis = List[str]


class TransformRecipe:
    dimensions = [] # list of results of expressions
    assignment_sequence = [] # argument dimension, divided by (multipliers)
    init_reshape_sequence = []
    


def parse_expression(expression) -> Tuple[Set[str], List[CompositeAxis]]:
    '''
    Parses an indexing expression (for a single tensor).
    Checks uniqueness of names, checks usage of '...'
    Returns set of all used identifiers and a list of axis groups
    '''
    identifiers = set('')
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
        if char in '().,':
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


def get_axes_names(composite_axis_name):
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

    # TODO add and delete dumny axes, add dots
    if identifiers_left != identifiers_rght:
        raise RuntimeError('Set of identifiers are different')

    # parsing all dimensions to find out lengths
    known_lengths = {}
    def update_axis_length(axis_name, axis_length):
        assert axis_length > 0
        axis_length = int(axis_length)
        if axis_name in known_lengths:
            assert axis_length == known_lengths[axis_name]
        else:
            known_lengths[axis_name] = axis_length

    # parsing explicitly passed sizes
    for axis, axis_length in axes_lengths.items():
        elementary_axes = get_axes_names(axis)
        if len(elementary_axes) == 1 and isinstance(axis_length, int):
            assert 'a' <= axis <= 'z'
            update_axis_length(elementary_axes[0], axis_length)
        else:
            assert len(elementary_axes) == len(axis_length), [elementary_axes, axis_length]
            for c, v in zip(axis, axis_length):
                if c != '_':
                    assert 'a' <= c <= 'z'
                    update_axis_length(c, v)

    # inferring rest of sizes from arguments
    assert len(composite_axes_left) == len(tensor.shape)
    for composite_axis, size in zip(composite_axes_left, tensor.shape):
        not_found = {axis for axis in composite_axis if axis not in known_lengths}
        found_product = 1
        for axis in composite_axis:
            if axis in known_lengths:
                found_product *= known_lengths[axis]
        if len(not_found) == 0:
            assert found_product == size
        else:
            assert len(not_found) == 1
            assert size % found_product == 0
            axis, = not_found
            known_lengths[axis] = size // found_product

    def compute_sizes_and_groups(composite_axes, known_sizes):
        axes_sizes = []
        groups_sizes = []
        for group in composite_axes:
            product = 1
            for name in group:
                axes_sizes.append(known_sizes[name])
                product *= known_sizes[name]
            groups_sizes.append(product)
        return axes_sizes, groups_sizes

    axes_sizes_left, group_sizes_left = compute_sizes_and_groups(
        composite_axes_left, known_sizes=known_lengths)
    axes_sizes_rght, group_sizes_rght = compute_sizes_and_groups(
        composite_axes_rght, known_sizes=known_lengths)

    def compute_matching(seq_left, seq_rght):
        # flatten dimensions and setup ordering
        l = list(itertools.chain(*seq_left))
        r = list(itertools.chain(*seq_rght))
        return [l.index(x) for x in r]

    matching = compute_matching(composite_axes_left, composite_axes_rght)
    assert list(group_sizes_left) == list(tensor.shape)
    assert isinstance(tensor, torch.Tensor)
    return tensor.reshape(axes_sizes_left).permute(matching).reshape(group_sizes_rght)
