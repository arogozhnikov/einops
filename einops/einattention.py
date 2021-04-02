from typing import Optional, Union, List, Dict, Set, Tuple
from .parsing import _ellipsis, ParsedExpression
from . import EinopsError, rearrange
from collections import namedtuple

Axis = namedtuple("axis", "type value")
ellipsis_axis = Axis(2, 0)
ellipsis_axis_parenthesized = Axis(2, 1)
named_axis_type = 1
anonymous_axis_type = 2

_parenthesized_ellipsis = "(" + _ellipsis + ")"


def parse_tensor_pattern_syntax(pattern) -> List[List[str]]:
    """
    Does not check anything about correctness, only if identifiers are possible and structure is correct
    :param pattern: e.g. ""x y323_w () (w 12 1) 1 ... (2 ... *) 4"
    parenthesized and non parenthesized ellipsis are different symbols, ones
    :return: [["x"], ["y323_w"], [], ["w", "12", "1"], ["1"], ["..."], ["2", "(...)", "*"], ["4"]]
    """
    parsed_pattern = []
    if '.' in pattern:
        if str.count(pattern, '...') != 1 or str.count(pattern, '.') != 3:
            raise EinopsError(
                'Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
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
        is_axis_name, reason = ParsedExpression.check_axis_name(x, return_reason=True)
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


def parse_tensor_pattern_advanced(
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
    :param tensor_pattern: output of parse_tensor_pattern_syntax
    :param axis2axis_id:
    :param allow_new_axes:
    :param check_has_axes_once:
    :return:
    """
    identifiers = set()
    new_structure: List[List[Axis]] = []
    for group in parsed_tensor_pattern:
        new_group = []
        for axis in group:
            if axis == _ellipsis or axis == _parenthesized_ellipsis:
                raise EinopsError("Duplicate axis name: {}".format(axis))

            if str.isnumeric(axis):
                if not allow_anonymous_axes:
                    raise EinopsError(f"Anonymous axis {axis} is not allowed")
                axis_length = int(axis)
                if axis_length <= 1:
                    raise EinopsError(f"Axis length of {axis_length} was not recognized")

                new_group.append(Axis(anonymous_axis_type, axis_length))
            else:
                if check_unique and axis in identifiers:
                    raise EinopsError("Duplicate axis name: '{}'".format(axis))
                identifiers.add(axis)

                if axis not in axis2axis_id:
                    if not allow_new_axes:
                        raise EinopsError("Unexpected axis name: {}".format(axis))
                    axis2axis_id[axis] = len(axis2axis_id)
                new_group.append(Axis(named_axis_type, axis2axis_id[axis]))

        new_structure.append(new_group)
    return identifiers, new_structure


def infer_and_check_sizes(tensor_shape: List[int], tensor_structure: List[List[Axis]], sizes: List[Union[int, None]]):
    # TODO introduce shortcut
    # if all(len(group) == 1 for group in tensor_structure):
    #     return []

    # shape when decomposed to elementary axes
    elementary_shape: List[int] = []
    for dimension, axes in zip(tensor_shape, tensor_structure):
        known_product = 1
        unknown_axes = []
        for axis in axes:
            if axis.type == named_axis_type:
                axis_size = sizes[axis.value]
                if axis_size is None:
                    unknown_axes.append(axis.value)
                else:
                    known_product *= axis_size
            else:
                assert axis.type == anonymous_axis_type
                known_product *= axis.value

        if len(unknown_axes) > 2:
            raise EinopsError(f"Could not infer sizes for axes simultaneously: {unknown_axes}")
        if len(unknown_axes) == 1:
            if dimension % known_product != 0:
                raise EinopsError(f"Dimension of length {dimension} is not divisible by {known_product}")
            sizes[unknown_axes[0]] = dimension // known_product
        else:
            if dimension != known_product:
                raise EinopsError(f"Sizes mismatch:  {dimension} != {known_product}")
        # now cycle again as all sizes are figured out
        for axis in axes:
            if axis.type == named_axis_type:
                elementary_shape.append(sizes[axis.value])
            else:
                elementary_shape.append(axis.value)

    return elementary_shape


def update_structure(structure: List[List[Axis]], old_axis_id, new_axis_id):
    result: List[List[Axis]] = []
    for input_group in structure:
        result_group = []
        for axis in input_group:
            if axis.type == named_axis_type and axis.value == old_axis_id:
                result_group.append(Axis(named_axis_type, new_axis_id))
            else:
                result_group.append(axis)
        result.append(result_group)
    return result


# class EinAttentionRecipe_nonfinished:
#     def __init__(self, pattern, **axis_lengths: int):
#         self.axis2axis_id: Dict[str, int] = {
#             '*': 0,
#             '*-axis in kq': 1,
#             '*-axis in v': 2,
#             # others will be filled along the way
#         }
#         result, right_part = pattern.split('<-')
#         q_pattern, kv_pattern = right_part.split(',')
#
#         q_pattern = parse_tensor_pattern_syntax(q_pattern)
#         kv_pattern = parse_tensor_pattern_syntax(kv_pattern)
#         q_identifiers, q_structure = parse_tensor_pattern_advanced(q_pattern,
#             axis2axis_id=self.axis2axis_id, allow_new_axes=True, allow_anonymous_axes=False)
#         kv_identifiers, kv_structure = parse_tensor_pattern_advanced(kv_pattern,
#             axis2axis_id=self.axis2axis_id, allow_new_axes=True, allow_anonymous_axes=False)
#         r_identifiers, r_structure = parse_tensor_pattern_advanced(result,
#             axis2axis_id=self.axis2axis_id, allow_new_axes=True, allow_anonymous_axes=False)
#
#         assert '*' in q_identifiers
#         assert '*' in kv_identifiers
#         assert q_identifiers == r_identifiers
#         star_name = 'star_dim'
#         assert star_name not in set.union(q_identifiers, kv_identifiers), "name star_name is not allowed for axis"
#
#
#
#
#         # replacing reference to * with references to q-version of kv-version
#         self.q_structure = update_structure(q_structure, 0, 1)
#         self.k_structure = update_structure(kv_structure, 0, 1)
#         self.v_structure = update_structure(kv_structure, 0, 2)
#         self.r_structure = update_structure(r_structure, 0, 2)
#
#         self.axis_lengths = [None] * len(self.axis2axis_id)
#         for axis_name, axis_length in axis_lengths.items():
#             self.axis_lengths[self.axis2axis_id[axis_name]] = axis_length
#
#         q_structure_flat = sum(self.q_structure, [])
#         k_structure_flat = sum(self.k_structure, [])
#         id2position: Dict[int] = {}
#         tensordot_left = []
#         tensordot_right = []
#
#         batch_like_ids = set.intersection(set(q_structure_flat), set(k_structure_flat))
#         def parse_positions(structure_flat):
#             batch_like_positions = []
#             other_positions = []
#             star_position = -1
#             for position, axis in enumerate(structure_flat):
#                 if axis in batch_like_ids:
#                     batch_like_positions.append(position)
#                 elif axis in [1, 2]:
#                     star_position = position
#                 else:
#                     assert axis != 0
#                     other_positions.append(axis)
#
#             assert star_position != -1
#             assert len(batch_like_positions) + len(other_positions) + 1 == len(structure_flat)
#             return batch_like_positions, other_positions, [star_position]
#
#         q_batch, q_other, q_star = parse_positions(q_structure_flat)
#         k_batch, k_other, k_star = parse_positions(k_structure_flat)
#         r_batch, r_other, r_star = parse_positions(sum(self.r_structure, []))
#
#         self.q_transposition = q_batch + q_other + q_star
#         self.k_transposition = k_batch + k_star + k_other
#         self.v_transposition = k_batch + k_other + k_star
#         self.r_transposition_inv = r_batch + r_other + r_star
#
#
#
#         for position, q_id in enumerate(q_structure_flat):
#             id2position[q_id.value] = position
#         matched_ids: Set[int] = set()
#         for k_position, k_id in enumerate(k_structure_flat):
#             if k_id in id2position:
#                 tensordot_left.append(id2position[k_id.value])
#                 tensordot_right.append(k_position)
#                 matched_ids.add(k_id.value)
#         self.qk_matched_axes = [tensordot_left, tensordot_right]
#         self.attention_axes = []
#         self.softmaxed_axes = []
#         self.result_axes = []
#         for q_id in q_structure_flat:
#             if q_id not in matched_ids:
#                 self.attention_axes.append(q_id)
#                 self.result_axes.append(q_id)
#         for k_id in k_structure_flat:
#             if k_id not in matched_ids:
#                 self.attention_axes.append(k_id)
#                 self.softmaxed_axes.append(len(self.attention_axes))
#
#         for v_position, v_id in enumerate(sum(self.v_structure, [])):
#             if v_id in matched_ids:
#
#
#         result_axes = []
#
#
#
#
#
#     def apply(self, query, key, value):
#         sizes = self.axis_lengths.copy()
#
#         query, key, value = [
#             x.reshape(infer_and_check_sizes(query.shape, self.q_structure, sizes=sizes))
#             for x in [query, key, value]
#         ]
#
#
#         attention = key.tensordot(query, self.qk_matched_axes).softmax(self.softmaxed_axes)
#
#
#
#
#
#
#
#
#
# def einattention(pattern, q, k, v, attention_mask=None, dropout=0.5):
#     pass


class EinAttentionRecipe:
    def __init__(self, pattern, **axis_lengths: int):
        star_name_kq = 'star_dim_kq'
        star_name_v = 'star_dim_v'

        self.axis2axis_id: Dict[str, int] = {
            '*': 0,
            star_name_kq: 1,
            star_name_v: 2,
            # others will be filled along the way
        }
        result, right_part = pattern.split('<-')
        q_part, kv_part = right_part.split(',')

        q_pattern = parse_tensor_pattern_syntax(q_part)
        kv_pattern = parse_tensor_pattern_syntax(kv_part)
        r_pattern = parse_tensor_pattern_syntax(result)
        settings = dict(axis2axis_id=self.axis2axis_id, allow_new_axes=True, allow_anonymous_axes=False)
        q_identifiers, q_structure = parse_tensor_pattern_advanced(q_pattern, **settings)
        kv_identifiers, kv_structure = parse_tensor_pattern_advanced(kv_pattern, **settings)
        r_identifiers, r_structure = parse_tensor_pattern_advanced(r_pattern, **settings)
        assert q_identifiers == r_identifiers
        assert star_name_kq not in set.union(q_identifiers, kv_identifiers)
        assert star_name_v not in set.union(q_identifiers, kv_identifiers)
        batch_identifiers = [i for i in kv_identifiers if i in q_identifiers and i != '*']

        self.axis_lengths = [None] * len(self.axis2axis_id)
        for axis_name, axis_length in axis_lengths.items():
            self.axis_lengths[self.axis2axis_id[axis_name]] = axis_length

        # replacing reference to * with references to q-version of kv-version
        self.q_structure = update_structure(q_structure, 0, 1)
        self.k_structure = update_structure(kv_structure, 0, 1)
        self.v_structure = update_structure(kv_structure, 0, 2)
        self.r_structure = update_structure(r_structure, 0, 2)

        def make_expression(left_part, identifiers, star_name, star_last):
            assert '*' in identifiers
            batch_part = ' '.join(batch_identifiers)
            seq_part = ' '.join(x for x in identifiers if x not in batch_identifiers and x != '*')
            result_identifiers = [x if x != '*' else star_name for x in identifiers]
            if star_last:
                right_part = f"({batch_part}) ({seq_part}) {star_name}"
            else:
                right_part = f"({batch_part}) {star_name} ({seq_part})"
            return f"{left_part.replace('*', star_name)} -> {right_part}", result_identifiers

        self.q_formula = make_expression(q_part, q_identifiers, star_name_kq, star_last=True)
        self.k_formula = make_expression(kv_part, kv_identifiers, star_name_kq, star_last=False)
        self.v_formula = make_expression(kv_part, kv_identifiers, star_name_v, star_last=True)
        r_pattern, r_identifiers = make_expression(result, r_identifiers, star_name_v, star_last=True)
        r_pattern = '->'.join(r_pattern.split('->')[::-1])
        self.r_formula = r_pattern, r_identifiers

    def forward(self, q, k, v):
        sizes = self.axis_lengths.copy()

        infer_and_check_sizes(q.shape, self.q_structure, sizes=sizes)
        infer_and_check_sizes(k.shape, self.k_structure, sizes=sizes)
        infer_and_check_sizes(v.shape, self.v_structure, sizes=sizes)

        def local_rearrange(tensor, formula):
            pattern, axes_identifiers = formula
            axes_sizes = {axis: sizes[self.axis2axis_id[axis]] for axis in axes_identifiers}
            return rearrange(tensor, pattern, **axes_sizes)

        q = local_rearrange(q, self.q_formula)
        k = local_rearrange(k, self.k_formula)
        v = local_rearrange(v, self.v_formula)

        logattention = q.bmm(k)
        attention = logattention.softmax(axis=-1)

        result = attention.bmm(v)
        return local_rearrange(result, self.r_formula)
