from typing import List, Dict, Set, Tuple, Optional

import torch.nn

from . import EinopsError
from .parsing import _ellipsis, ParsedExpression


@torch.jit.script
class Axis:
    named_axis_type = 1
    anonymous_axis_type = 2

    def __init__(self, kind: int, value: int):
        self.kind: int = kind
        self.value: int = value

    def __repr__(self):
        return f'Axis(kind={self.kind}, value={self.value})'


_parenthesized_ellipsis = "(" + _ellipsis + ")"


def parse_tensor_pattern_lexical(pattern) -> List[List[str]]:
    """
    Does not check anything about correctness, only if identifiers are possible and structure is correct
    :param pattern: e.g. ""x y323_w () (w 12 1) 1 ... (2 ... *) 4"
    parenthesized and non parenthesized ellipsis are different symbols:
    :return: [["x"], ["y323_w"], [], ["w", "12", "1"], ["1"], ["..."], ["2", "(...)", "*"], ["4"]]
    """
    parsed_pattern = []
    if '.' in pattern:
        if str.count(pattern, '...') != 1 or str.count(pattern, '.') != 3:
            raise EinopsError('Expression may contain dots only inside ellipsis (...); one ellipsis for tensor')
        pattern = pattern.replace('...', _ellipsis)

    bracket_group = []
    within_brackets = False

    def add_axis_name(x):
        if x == "":
            return
        if x == "1":
            if not within_brackets:
                parsed_pattern.append([])
                return
        is_ellipsis = False
        if x == _ellipsis:
            is_ellipsis = True
            if within_brackets:
                # parenthesized ellipsis
                x = _parenthesized_ellipsis

        is_number = str.isalnum(x)
        is_star = '*'
        is_axis_name, reason = ParsedExpression.check_axis_name_return_reason(x)
        if not (is_number or is_axis_name or is_ellipsis or is_star):
            raise EinopsError('Invalid axis identifier: {}\n{}'.format(x, reason))
        if within_brackets:
            bracket_group.append(x)
        else:
            parsed_pattern.append([x])

    current_identifier = ""
    for char in pattern:
        if char in '() ':
            add_axis_name(current_identifier)
            current_identifier = ""
            if char == '(':
                if within_brackets:
                    raise EinopsError("Axis composition is one-level (brackets inside brackets not allowed)")
                within_brackets = True
            elif char == ')':
                if not within_brackets:
                    raise EinopsError('Brackets are not balanced')
                parsed_pattern.append(bracket_group)
                bracket_group = []
                within_brackets = False
        elif str.isalnum(char) or char in ['_', '*', _ellipsis]:
            current_identifier += char
        else:
            raise EinopsError("Unknown character '{}'".format(char))

    if within_brackets:
        raise EinopsError('Imbalanced parentheses in pattern: "{}"'.format(pattern))
    add_axis_name(current_identifier)
    return parsed_pattern


def parse_tensor_structure(
        parsed_tensor_pattern: List[List[str]],
        axis2axis_id: Dict[str, int],
        allow_new_axes: bool,
        check_unique: bool = True,
        allow_anonymous_axes: bool = True,
) -> Tuple[
    Set[str],
    List[List[Axis]],
]:
    """
    :param parsed_tensor_pattern: output of parse_tensor_pattern_syntax
    :param axis2axis_id: name2length, known values
    :param allow_new_axes: allow axes not in axis2axis_id
    :param check_unique: restrict duplicate elementary axes
    :param allow_anonymous_axes: allow axes like 3, 5. In reduce patterns: left is ''
    """
    identifiers = set()
    new_structure: List[List[Axis]] = []
    for group in parsed_tensor_pattern:
        new_group = []
        for axis in group:
            if axis == _ellipsis or axis == _parenthesized_ellipsis:
                raise EinopsError("Parenthesis is not supported")

            if str.isnumeric(axis):
                if not allow_anonymous_axes:
                    raise EinopsError(f"Anonymous axis {axis} is not allowed")
                axis_length = int(axis)
                if axis_length <= 1:
                    raise EinopsError(f"Axis length of {axis_length} was not recognized")

                new_group.append(Axis(Axis.anonymous_axis_type, axis_length))
            else:
                if check_unique and axis in identifiers:
                    raise EinopsError("Duplicate axis name: '{}'".format(axis))
                identifiers.add(axis)

                if axis not in axis2axis_id:
                    if not allow_new_axes:
                        raise EinopsError("Unexpected axis name: {}".format(axis))
                    axis2axis_id[axis] = len(axis2axis_id)
                new_group.append(Axis(Axis.named_axis_type, axis2axis_id[axis]))

        new_structure.append(new_group)
    return identifiers, new_structure


def infer_and_check_sizes(
        tensor_shape: List[int],
        tensor_structure: List[List[Axis]],
        sizes: List[Optional[int]],
        source_name: str,
        # lines below required to pass torch.jit
        named_axis_type: int = Axis.named_axis_type,
        anonymous_axis_type: int = Axis.anonymous_axis_type,
):
    """
    Accepts only named and anonymous axes. Infers missing ones, checks divisibility otherwise

    Example:
    tensor_shape: (3, 4, 6)
    sizes: [2, None, None, None, None, 3, 4, 5]
    tensor_structure: [ [Anon(3), Anon(1)], [Named(0), Named(1), Anon(2)], [Named[2]] ]

    This will pass successfully, resulting sizes (sizes[1] and sizes[2] were filled from the data):
    sizes: [2,    1,    6, None, None, 3, 4, 5]
    """

    # shape when decomposed to elementary axes
    elementary_shape: List[int] = []
    for dimension, axes in zip(tensor_shape, tensor_structure):
        known_product = 1
        unknown_axes: List[int] = []
        for axis in axes:
            if axis.kind == named_axis_type:
                axis_size = sizes[axis.value]
                if axis_size is None:
                    unknown_axes.append(axis.value)
                else:
                    known_product *= axis_size
            else:
                assert axis.kind == anonymous_axis_type
                known_product *= axis.value

        if len(unknown_axes) > 2:
            raise EinopsError(f"Could not infer sizes for axes simultaneously in {source_name}: {unknown_axes}")
        if len(unknown_axes) == 1:
            if dimension % known_product != 0:
                raise EinopsError(
                    f"Dimension of length {dimension} is not divisible by {known_product} in {source_name}")
            sizes[unknown_axes[0]] = dimension // known_product
        else:
            if dimension != known_product:
                raise EinopsError(f"Sizes mismatch in {source_name}: {dimension} != {known_product}")
        # now cycle again as all sizes are figured out
        for axis in axes:
            if axis.kind == named_axis_type:
                x = sizes[axis.value]
                assert x is not None
                elementary_shape.append(x)
            else:
                x = axis.value
                assert isinstance(x, int)
                elementary_shape.append(x)

    return elementary_shape


def update_structure(structure: List[List[Axis]], old_axis_id, new_axis_id) -> List[List[Axis]]:
    result: List[List[Axis]] = []
    for input_group in structure:
        result_group = []
        for axis in input_group:
            if axis.kind == Axis.named_axis_type and axis.value == old_axis_id:
                result_group.append(Axis(Axis.named_axis_type, new_axis_id))
            else:
                result_group.append(axis)
        result.append(result_group)
    return result


def find_permutation(source: List[int], dest: List[int]) -> List[int]:
    name2source_order: Dict[int, int] = {}
    for order, x in enumerate(source):
        name2source_order[x] = order
    return [name2source_order[x] for x in dest]


def structure_axes_to_ids(structure: List[List[Axis]]) -> List[List[int]]:
    result_structure: List[List[int]] = []
    for group in structure:
        result_group = []
        for axis in group:
            assert axis.kind == Axis.named_axis_type
            result_group.append(axis.value)
        result_structure.append(result_group)
    return result_structure


@torch.jit.script
class GroupingRecipe:
    def __init__(
            self,
            structure: List[List[int]],
            batch_identifiers: List[int],
            star_axis_id: int,
            star_last: bool
    ):
        self.ids_order_before_transpose: List[int] = []
        self.structure: List[List[int]] = structure
        for seq in structure:
            self.ids_order_before_transpose.extend(seq)

        assert star_axis_id in self.ids_order_before_transpose, (self.ids_order_before_transpose, star_axis_id)

        self.seq_identifiers: List[int] = []
        for x in self.ids_order_before_transpose:
            if x not in batch_identifiers and x != star_axis_id:
                self.seq_identifiers.append(x)

        if star_last:
            groups = [batch_identifiers, self.seq_identifiers, [star_axis_id]]
        else:
            groups = [batch_identifiers, [star_axis_id], self.seq_identifiers]
        self.lengths = [len(x) for x in groups]
        self.ids_order_after_transposition: List[int] = groups[0] + groups[1] + groups[2]

        self.fwd_transposition = find_permutation(self.ids_order_before_transpose, self.ids_order_after_transposition)
        self.bwd_transposition = find_permutation(self.ids_order_after_transposition, self.ids_order_before_transpose)

    def forward(self, tensor, axis_id2size: List[int]):
        shape: List[int] = []
        for i in self.ids_order_before_transpose:
            shape.append(axis_id2size[i])
        tensor = tensor.reshape(shape).permute(self.fwd_transposition)
        tensor = tensor.flatten(0, self.lengths[0] - 1).flatten(1, self.lengths[1]).flatten(2, -1)
        assert len(tensor.shape) == 3
        return tensor

    def backward(self, tensor, axis_id2size: List[int]):
        shape: List[int] = []
        for i in self.ids_order_after_transposition:
            shape.append(axis_id2size[i])
        tensor = tensor.reshape(shape).permute(self.bwd_transposition)
        final_shape: List[int] = []
        for group in self.structure:
            axis_size: int = 1
            for axis_id in group:
                axis_size *= axis_id2size[axis_id]
            final_shape.append(axis_size)
        tensor = tensor.reshape(final_shape)
        return tensor


class EinAttention(torch.nn.Module):

    def __init__(self,
                 pattern,
                 /, *,
                 mask_dimensions: Optional[str] = None,
                 **axis_lengths: int):
        super().__init__()
        star_name_kq = 'star_dim_kq'
        star_name_v = 'star_dim_v'

        self.axis2axis_id: Dict[str, int] = {
            '*': 0,
            star_name_kq: 1,
            star_name_v: 2,
            # others will be added along the way
        }
        result, right_part = pattern.split('<-')
        q_part, kv_part = right_part.split(',')

        q_pattern = parse_tensor_pattern_lexical(q_part)
        kv_pattern = parse_tensor_pattern_lexical(kv_part)
        r_pattern = parse_tensor_pattern_lexical(result)
        settings = dict(axis2axis_id=self.axis2axis_id, allow_new_axes=True, allow_anonymous_axes=False)
        q_identifiers, q_structure = parse_tensor_structure(q_pattern, **settings)
        kv_identifiers, kv_structure = parse_tensor_structure(kv_pattern, **settings)
        r_identifiers, r_structure = parse_tensor_structure(r_pattern, **settings)
        if q_identifiers != r_identifiers:
            diff = set.symmetric_difference(q_identifiers, r_identifiers)
            raise EinopsError(f'Query and result parts should have identical axes in pattern "{pattern}": {diff}')
        assert q_identifiers == r_identifiers, 'Idenfiers'
        assert star_name_kq not in set.union(q_identifiers, kv_identifiers), f"don't use {star_name_kq} in pattern"
        assert star_name_v not in set.union(q_identifiers, kv_identifiers), f"don't use {star_name_v} in pattern"
        batch_identifiers = [i for i in kv_identifiers if i in q_identifiers and i != '*']

        self.axis_lengths: List[Optional[int]] = [None] * len(self.axis2axis_id)
        for axis_name, axis_length in axis_lengths.items():
            self.axis_lengths[self.axis2axis_id[axis_name]] = axis_length

        # replace reference to * with references to q-version or kv-version.
        self.q_structure = update_structure(q_structure, 0, 1)
        self.k_structure = update_structure(kv_structure, 0, 1)
        self.v_structure = update_structure(kv_structure, 0, 2)
        self.r_structure = update_structure(r_structure, 0, 2)

        batch_ids = [self.axis2axis_id[i] for i in batch_identifiers]
        ids = structure_axes_to_ids
        self.q_formula = GroupingRecipe(
            ids(self.q_structure), batch_ids, self.axis2axis_id[star_name_kq], star_last=True)
        self.k_formula = GroupingRecipe(
            ids(self.k_structure), batch_ids, self.axis2axis_id[star_name_kq], star_last=False)
        self.v_formula = GroupingRecipe(
            ids(self.v_structure), batch_ids, self.axis2axis_id[star_name_v], star_last=True)
        self.r_formula = GroupingRecipe(
            ids(self.r_structure), batch_ids, self.axis2axis_id[star_name_v], star_last=True)

        if mask_dimensions is not None:
            self.mask_all_dimensions = [batch_ids, self.q_formula.seq_identifiers, self.k_formula.seq_identifiers]
            all_mask_identifiers_flat = sum(self.mask_all_dimensions, [])
            passed_mask_dimensions = [self.axis2axis_id[axis] for axis in mask_dimensions.split(' ')]
            self.mask_transposition = [
                passed_mask_dimensions.index(dim)
                for dim in all_mask_identifiers_flat
                if dim in passed_mask_dimensions
            ]
            self.mask_indexer = [
                slice(None, None, None) if dim in passed_mask_dimensions else None
                for dim in all_mask_identifiers_flat
            ]
        else:
            self.mask_all_dimensions = None
            self.mask_transposition = None
            self.mask_indexer = None

    def forward(self, q, k, v, mask=None):
        sizes: List[Optional[int]] = self.axis_lengths.copy()

        print(f'{sizes=}, {q.shape=}, {self.q_structure=}')
        infer_and_check_sizes(q.shape, self.q_structure, sizes=sizes, source_name='query part')
        infer_and_check_sizes(k.shape, self.k_structure, sizes=sizes, source_name='key part')
        infer_and_check_sizes(v.shape, self.v_structure, sizes=sizes, source_name='value part')

        sizes: List[int] = [-1234567890 if x is None else x for x in sizes]

        q = self.q_formula.forward(q, sizes)
        k = self.k_formula.forward(k, sizes)
        v = self.v_formula.forward(v, sizes)

        logattention = q.bmm(k)
        if mask is not None:
            reshaped_mask = mask.permute(self.mask_transposition)[self.mask_indexer]
            logattention.view(self.mask_all_dimensions).add_(reshaped_mask)

        attention = logattention.softmax(-1)
        # TODO add dropout here

        result = attention.bmm(v)
        return self.r_formula.backward(result, sizes)
