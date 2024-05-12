"""
torch-only version for efficient production of multiple outputs from the same input,
while making all rearranges.
python 3.9+ because of typing.

implementation is a bit fragile, pin exact version if using.
Name isn't great, other names under consideration:
 - multilinear (... confusion with multilinearity),
 - mergedlinear
 - multiprojection
"""
from typing import Iterable

import torch
from torch.functional import F
from torch.nn import ModuleList, Parameter

from einops.einops import _product
from einops.layers.torch import Rearrange
from einops.parsing import ParsedExpression


def _split_into_groups(pattern: str) -> list[list[str]]:
    # does not differentiate composed and non-composed ellipsis
    result: list[list[str]] = []
    pattern_rest: str = pattern
    # check there is space before `(` and after `)` for proper style
    pattern_with_edges = f' {pattern} '
    msg = f'please add spaces before and after parenthesis in {pattern=}'
    assert pattern_with_edges.count(' (') == pattern_with_edges.count('('), msg
    assert pattern_with_edges.count(') ') == pattern_with_edges.count(')'), msg

    while True:
        if pattern_rest.startswith('('):
            i = pattern_rest.index(')')
            group, pattern_rest = pattern_rest[1:i], pattern_rest[i + 1:]
            assert '(' not in group, 'unbalanced brackets'
            result.append(group.split())
        elif '(' in pattern_rest:
            i = pattern_rest.index('(')
            ungrouped, pattern_rest = pattern_rest[:i], pattern_rest[i:]
            assert ')' not in ungrouped, 'unbalanced brackets'
            result.extend([[x] for x in pattern_rest.split()])
        else:
            # no more brackets, just parse the end
            result.extend([[x] for x in pattern_rest.split()])
            break
    return result


def _join_groups_to_pattern(groups: list[list[str]]) -> str:
    result = ''
    for group in groups:
        if len(group) == 1:
            result += f'{group[0]} '
        else:
            result += '(' + ' '.join(group) + ') '
    return result.strip()


def _assert_good_identifier(axis_label: str) -> None:
    valid, reason = ParsedExpression.check_axis_name_return_reason(
        axis_label, allow_underscore=False
    )
    assert valid, f'Bad {axis_label=}, {reason}'


def _get_name_for_anon_axis(disallowed_axes: Iterable[str], axis_len: int) -> str:
    prefix = 'c'
    while True:
        prefix += '_'
        axis_name = f'{prefix}{axis_len}'
        if axis_name not in disallowed_axes:
            return axis_name


def _process_input_pattern(input_pattern) -> tuple[Rearrange, list[str], int]:
    """
    examples of input patterns: 'a 1 (2 3 b) ()'  'c d e f 9'
    does not support ellipsis, and tagging of variables, like c=4
    """
    groups = _split_into_groups(input_pattern)

    all_identifiers = [el.partition('=')[0] for group in groups for el in group if not str.isnumeric(el)]
    assert len(all_identifiers) == len(set(all_identifiers)), f"duplicate names in {input_pattern=}"

    batch_axes = []
    input_axes2size = {}
    named_groups = []
    for group in groups:
        named_group = []
        for axis in group:
            if '=' in axis:
                axis_name, _, axis_len_str = axis.partition('=')
                axis_len = int(axis_len_str)
                assert axis_len > 0, axis
                _assert_good_identifier(axis_name)
                input_axes2size[axis_name] = axis_len
                named_group.append(axis_name)

            elif str.isnumeric(axis):
                axis_len = int(axis)
                assert axis_len > 0, f'{axis_len=}'
                axis_name = _get_name_for_anon_axis(all_identifiers, axis_len)
                input_axes2size[axis_name] = axis_len
                named_group.append(axis_name)
            else:
                _assert_good_identifier(axis)
                batch_axes.append(axis)
                named_group.append(axis)

        named_groups.append(named_group)

    init_reordering_pattern = _join_groups_to_pattern(named_groups)
    init_reordering_pattern += ' -> ' + _join_groups_to_pattern([[x] for x in batch_axes] + [list(input_axes2size)])
    total_input_size = _product(list(input_axes2size.values()))

    return Rearrange(init_reordering_pattern, **input_axes2size), batch_axes, total_input_size


def _process_output_pattern(output_pattern, batch_axes) -> tuple[Rearrange, int]:
    groups = _split_into_groups(output_pattern)

    all_identifiers = [el.partition('=')[0] for group in groups for el in group if not str.isnumeric(el)]
    assert len(all_identifiers) == len(set(all_identifiers)), f"duplicate names in {output_pattern=}"

    output_axis2size = {}
    named_groups = []
    for group in groups:
        named_group = []
        for axis in group:
            assert '=' not in axis, f'wrong identifier {axis=}, no names in outputs'
            if str.isnumeric(axis):
                axis_len = int(axis)
                assert axis_len > 0, f'{axis_len=}'
                axis_name = _get_name_for_anon_axis(all_identifiers, axis_len)
                output_axis2size[axis_name] = axis_len
                named_group.append(axis_name)
            else:
                assert axis in batch_axes, f'unknown axis in output, allowed only {batch_axes=}'
                named_group.append(axis)

        named_groups.append(named_group)

    reordering_pattern = _join_groups_to_pattern([[x] for x in batch_axes] + [[*output_axis2size]])
    reordering_pattern += ' -> ' + _join_groups_to_pattern(named_groups)
    total_output_size = _product(list(output_axis2size.values()))
    return Rearrange(reordering_pattern, **output_axis2size), total_output_size


class EinSplit(torch.nn.Module):
    def __init__(self, input_pattern: str):
        super().__init__()
        """all dimensions should be provided in-place"""
        self.input_pattern = input_pattern
        self.outputs: list[tuple] = []
        # intermediate parsing results

        # parsed = ParsedExpression(input_pattern)
        # if parsed.has_ellipsis:
        #     raise RuntimeError("no support for ellipsis so far")
        # self._required_identifiers = parsed.identifiers

        self._in_rearrange, self.batch_axes, self._total_input_size = \
            _process_input_pattern(input_pattern)
        self._out_rearranges = ModuleList([])
        self._out_sizes = []
        self.linear = None  # set after create_weights

        self.weight = Parameter(torch.empty([0, self._total_input_size]))
        self.bias = Parameter(torch.empty([0]))
        self.bias_mask = Parameter(torch.empty([0], dtype=torch.bool), requires_grad=False)

    def add_output(self, pattern: str, init: str = 'xavier_normal', bias: bool = True) -> int:
        """ returns index in output list """
        idx = len(self.outputs)
        out_rearrange, out_total_size = \
            _process_output_pattern(pattern, batch_axes=self.batch_axes)
        self.outputs.append((pattern, init, bias))
        self._out_sizes.append(out_total_size)
        self._out_rearranges.append(out_rearrange)

        W = self.weight.new_zeros(out_total_size, self._total_input_size)
        b = self.bias.new_zeros(out_total_size)
        b_mask = self.bias_mask.new_full(size=(out_total_size,), fill_value=int(bias), dtype=torch.bool)

        if init == 'xavier_normal':
            torch.nn.init.xavier_normal_(W)  # bias is zero
        elif init == 'zeros':
            torch.nn.init.zeros_(W)  # bias is zero
        else:
            raise ValueError(f'Unknown {init=}')

        with torch.no_grad():
            self.weight = Parameter(torch.concatenate([self.weight, W]))
            self.bias = Parameter(torch.concatenate([self.bias, b]))
            self.bias_mask = Parameter(torch.concatenate([self.bias_mask, b_mask]), requires_grad=False)

        return idx

    def __repr__(self):
        output = f"EinSplit({self.input_pattern})"
        for i, (pattern, init, bias, *_) in self.outputs:
            output += f'\n + output {i}:   {pattern}; {bias=}, {init=}'
        return output

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        merged = F.linear(self._in_rearrange(x), self.weight, self.bias * self.bias_mask)
        split = torch.split(merged, self._out_sizes, dim=-1)
        return [
            rearr_out(x) for rearr_out, x in zip(self._out_rearranges, split)
        ]

