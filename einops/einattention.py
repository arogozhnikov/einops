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
    List[str],
    List[List[Axis]],
]:
    """
    :param parsed_tensor_pattern: output of parse_tensor_pattern_syntax
    :param axis2axis_id: name2length, known values
    :param allow_new_axes: allow axes not in axis2axis_id
    :param check_unique: restrict duplicate elementary axes
    :param allow_anonymous_axes: allow axes like 3, 5. In reduce patterns: left is ''
    """
    identifiers = list()
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
                identifiers.append(axis)

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
    # force type for torch.
    # if this type hint is not provided, torch can recognize var as List[NoneType] and fail.

    axis_lengths: List[Optional[int]]

    def __init__(
            self,
            pattern: str,
            /,
            *,
            logshift_param: Optional[str] = None,
            logshift_forward: Optional[str] = None,
            normalize: bool = True,
            **axis_lengths: int
    ):
        """
        Flexible attention layer.

        Covered cases:
        1d / 2d / 3d / etc attention
        batched and single-instance attention
        strided attention
        window-ed attention

        Layer does not deal with pre- or post- normalizations and linear projections.

        Layer is scriptable, relies on bmm internally.

        Most important change is introduction of wildcard dimension (*) that is used as an embedding dimension
        of attention. This could be a keyword-axis, but its behavior is too different: this axis is matched across k,q,v,
        but it can have one length in q and k, and different in v. That's rare, but worth considering.

        :param pattern: <result_pattern> <- <query_pattern>, <key_and_value_pattern>
            Note that key and value should have the same pattern, while query has a different pattern.

        :param logshift_param: creates an internal logshift (logit shift) parameter of provided pattern;
            parameter is zero-initialized and properly aligned to logits.
            Lengths of all axes participating in this pattern should be provided.

        :param logshift_forward: allows passing logshift (logit shift) during forward; allows complex masking on-the-fly
            layer provides a different mechanism for simpler masks.
            Two logshifts can be applied at the same time.

        :param axis_lengths: lengths of axes
        """
        super().__init__()
        star_name_kq = 'star_dim_kq'
        star_name_v = 'star_dim_v'

        self.normalize = normalize

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
            diff = set.symmetric_difference(set(q_identifiers), set(r_identifiers))
            raise EinopsError(f'Query and result parts should have identical axes in pattern "{pattern}": {diff}')
        assert q_identifiers == r_identifiers, 'Idenfiers'
        assert star_name_kq not in set.union(set(q_identifiers), set(kv_identifiers)), f"don't use {star_name_kq} in pattern"
        assert star_name_v not in set.union(set(q_identifiers), set(kv_identifiers)), f"don't use {star_name_v} in pattern"
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

        mask_axis_ids = batch_ids + self.q_formula.seq_identifiers + self.k_formula.seq_identifiers
        self.mask_all_axis_ids = mask_axis_ids
        axis_id2axis_name = {v: k for k, v in self.axis2axis_id.items()}
        self.logit_axes_names = [axis_id2axis_name[axis_id] for axis_id in mask_axis_ids]
        self.logit_axis_name2position = {name: p for p, name in enumerate(self.logit_axes_names)}

        if logshift_param is not None:
            # provided order does not matter, parameter has its own shape aligned for simple forwarding
            logshift_param_axes: List[str] = logshift_param.split()
            assert all(axis_name in self.logit_axes_names for axis_name in logshift_param_axes)

            param_shape = []
            for axis in self.logit_axes_names:
                if axis in logshift_param_axes:
                    axis_length = self.axis_lengths[self.axis2axis_id[axis]]

                    assert isinstance(axis_length, int), f"length of {axis} can't be {axis_length}"
                    assert axis_length > 0, f"length of {axis} can't be {axis_length}"
                    param_shape.append(axis_length)
                else:
                    param_shape.append(1)
            self.logshift_param = torch.nn.Parameter(torch.zeros(param_shape), requires_grad=True)
        else:
            self.logshift_param = None

        self.logshift_forward = logshift_forward

        if logshift_forward is not None:
            # ids of axes in logshift. Order is very important
            passed_mask_axis_ids = [self.axis2axis_id[axis] for axis in logshift_forward.split()]
            self.mask_transposition = [
                passed_mask_axis_ids.index(dim) for dim in mask_axis_ids if dim in passed_mask_axis_ids
            ]
            self.logshift_forward_added_dimensions = [
                i for i, dim in enumerate(mask_axis_ids) if (dim not in passed_mask_axis_ids)
            ]
            # more efficient way, but torch scripting does not understand it
            # self.logshift_forward_indexer = [
            #     slice(None, None, None) if dim in passed_mask_axis_ids else None for dim in mask_axis_ids
            # ]
        else:
            # set all to Nones so that indexer
            self.mask_transposition = None
            self.logshift_forward_added_dimensions = None

    def modify_logit_inplace(self, logit_all_dims):
        """
        users can override this function during subclassing to get some convenient masking
        for multiple cases.

        See an example below.
        """
        pass

    def example_modify_logit_inplace(self, logit_all_dims: torch.Tensor):
        # this one is questionable:
        # there is a simple hack to specify that some positions shouldn't be attended to
        # by putting NaNs in corresponding embeddings.
        # and filling positions with a number.
        # This is lazy! Better provide logshift during computations
        torch.nan_to_num(logit_all_dims, nan=-1000, out=logit_all_dims)

        # 1-d example of temporal ordering: query only from previous or current positions
        t_q, t_kv = self.get_logit_dimensions_grid(logit_all_dims, ['t_q', 't_kv'])

        causal_mask = (t_kv <= t_q)
        logit_all_dims.add_(causal_mask.float().mul(100))

        # 1-d example of stripe mask: query only from neighboring positions
        stripe_mask = abs(t_kv - t_q) <= 10
        logit_all_dims.add_(stripe_mask.float().mul(100))

        # 2-d example of causal ordering when generating pixel-by-pixel
        h_q, w_q, h_kv, w_kv = self.get_logit_dimensions_grid(logit_all_dims, ['h_q', 'w_q', 'h_kv', 'w_kv'])
        is_prev = (h_q > h_kv) | ((h_q == h_kv) & (w_q >= w_kv))
        logit_all_dims.grad(is_prev.float().mul(100))

        # etc.
        # it is important to track order of dimensions in logit_all_dims here,
        # as self.get_logit_dimensions_grid(...) will return them as in a specific order that
        # is stored in self.logit_axes_names

    def get_logit_dimensions_grid(self, logit: torch.Tensor, axes: List[str]) -> List[torch.Tensor]:
        """
        supplementary function to help with 'example_modify_logit_inplace'

        NB: returns floating tensors
        :param logit: logit tensor, only it's shape and device are used, not contents.
        :return: list of pseudo-1d tensors, each of them aligned to an axis in logit_all_dims
            each tensor has only one non-unitary dimension, elements along this dimension are 0, 1, ... dim - 1.
        """
        result: List[torch.Tensor] = []
        logit_shape: List[int] = logit.shape
        for axis_name in axes:
            dim = self.logit_axis_name2position[axis_name]
            length = logit_shape[dim]
            # using longs is very inefficient, but that's the only type that works for torch indexing
            x = torch.arange(length, device=logit.device, dtype=torch.long)
            shape = [1] * len(logit_shape)
            shape[dim] = logit_shape[dim]
            result.append(x.reshape(shape))

        return result

    def forward(self, q, k, v, logshift: Optional[torch.Tensor] = None):
        """
        q, k, v - input tensors in accordance to patterns
        return output tensor (according to output part in a pattern)
        """
        # to allow different batches have different sizes,
        # we take a copy. These sizes will be checked or filled for each tensor.
        # This step checks shapes and deducts axes lengths.
        sizes: List[Optional[int]] = [x for x in self.axis_lengths]

        infer_and_check_sizes(q.shape, self.q_structure, sizes=sizes, source_name='query part')
        infer_and_check_sizes(k.shape, self.k_structure, sizes=sizes, source_name='key part')
        infer_and_check_sizes(v.shape, self.v_structure, sizes=sizes, source_name='value part')

        sizes: List[int] = [-1234567890 if x is None else x for x in sizes]

        q = self.q_formula.forward(q, sizes)
        k = self.k_formula.forward(k, sizes)
        v = self.v_formula.forward(v, sizes)

        logattention = q.bmm(k)
        if self.normalize:
            # NB this will not work with FP16, only bf16 and float32
            logattention *= q.shape[-1] ** -0.5

        logattention_detailed_shape = [sizes[i] for i in self.mask_all_axis_ids]
        logattention_view = logattention.view(logattention_detailed_shape)

        if self.logshift_param is not None:
            logattention_view.add_(self.logshift_param)

        # add logshift based on transposition
        if self.mask_transposition is not None:
            assert isinstance(logshift, torch.Tensor), f'logshift of shape {self.logshift_forward} should be passed'
            reshaped_logshift = logshift.permute(self.mask_transposition)
            for dim in self.logshift_forward_added_dimensions:
                reshaped_logshift = reshaped_logshift.unsqueeze(dim)
            logattention_view.add_(reshaped_logshift)
        else:
            assert logshift is None, 'logshift_forward were not passed during layer construction'

        self.modify_logit_inplace(logit_all_dims=logattention_view)

        # turn on this check during debugging to ensure you keep editing the same tensor
        # assert torch.equal(logattention_view.reshape(logattention.shape), logattention)

        attention = logattention.softmax(-1)

        # TODO add dropout here?

        result = attention.bmm(v)
        return self.r_formula.backward(result, sizes)

    def __repr__(self) -> str:
        # TODO better layer representation
        return f'Einattention(...)'
