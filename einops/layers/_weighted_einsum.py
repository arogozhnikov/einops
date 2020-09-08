from einops import EinopsError
from einops.parsing import ParsedExpression, _ellipsis, AnonymousAxis
import warnings
import string
from ..einops import _product


def _report_axes(axes: set, report_message: str):
    if len(axes) > 0:
        raise EinopsError(report_message.format(axes))


class WeightedEinsumMixin:
    def __init__(self, pattern, weight_shape, bias_shape=None, **axes_lengths):
        """
        WeightedEinsum - Einstein summation with second argument being weight tensor.
        NB: it is an experimental API. RFC https://github.com/arogozhnikov/einops/issues/71

        Imagine taking einsum with two arguments, one of each input, and one - tensor with weights
        >>> einsum('time batch channel_in, channel_in channel_out -> time batch channel_out', input, weight)

        This layer manages weights for you after a minor tweaking
        >>> WeightedEinsum('time batch channel_in -> time batch channel_out', weight_shape='channel_in channel_out')
        But otherwise it is the same einsum.

        Simple linear layer with bias term (you have one like that in your framework)
        >>> WeightedEinsum('t b cin -> t b cout', weight_shape='cin cout', bias_shape='cout', cin=10, cout=20)
        Channel-wise multiplication (like one used in normalizations)
        >>> WeightedEinsum('t b c -> t b c', weight_shape='c', c=128)
        Separate dense layer within each head, no connection between different heads
        >>> WeightedEinsum('t b head cin -> t b head cout', weight_shape='head cin cout', ...)

        ... ah yes, you need to specify all dimensions of weight shape/bias shape in parameters.

        Good use cases:
        - when channel dimension is not last, use WeightedEinsum, not transposition
        - when need only within-group connections to reduce number of weights and computations
        - perfect as a part of sequential models

        Uniform He initialization is applied to weight tensor.

        Parameters
        :param pattern: transformation pattern, left side - dimensions of input, right side - dimensions of output
        :param weight_shape: axes of weight. Tensor od this shape is created, stored, and optimized in a layer
        :param bias_shape: axes of bias added to output.
        :param axes_lengths: dimensions of weight tensor
        """
        super().__init__()
        warnings.warn('WeightedEinsum is experimental feature. API can change in unpredictable and enjoyable ways',
                      FutureWarning)
        self.pattern = pattern
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.axes_lengths = axes_lengths

        left, right = pattern.split('->')
        left = ParsedExpression(left)
        right = ParsedExpression(right)
        weight = ParsedExpression(weight_shape)
        _report_axes(
            set.difference(right.identifiers, {*left.identifiers, *weight.identifiers}),
            'Unrecognized identifiers on the right side of WeightedEinsum {}'
        )

        if left.has_ellipsis or right.has_ellipsis or weight.has_ellipsis:
            raise EinopsError('Ellipsis is not supported in WeightedEinsum (right now)')
        if any(x.has_non_unitary_anonymous_axes for x in [left, right, weight]):
            raise EinopsError('Anonymous axes (numbers) are not allowed in WeightedEinsum')
        if '(' in weight_shape or ')' in weight_shape:
            raise EinopsError('Parenthesis is not allowed in weight shape')
        # TODO implement this
        if '(' in pattern or ')' in pattern:
            raise EinopsError('Axis composition/decomposition are not yet supported in einsum')
        for axis in weight.identifiers:
            if axis not in axes_lengths:
                raise EinopsError('Dimension {} of weight should be specified'.format(axis))
        _report_axes(
            set.difference(set(axes_lengths), {*left.identifiers, *weight.identifiers}),
            'Axes {} are not used in pattern',
        )
        _report_axes(
            set.difference(weight.identifiers, {*left.identifiers, *right.identifiers}),
            'Weight axes {} are redundant'
        )
        if len(weight.identifiers) == 0:
            warnings.warn('WeightedEinsum: weight has no dimensions (means multiplication by a number)')

        _weight_shape = [axes_lengths[axis] for axis, in weight.composition]
        # single output element is a combination of fan_in input elements
        _fan_in = _product([axes_lengths[axis] for axis, in weight.composition if axis not in right.identifiers])
        if bias_shape is not None:
            if not isinstance(bias_shape, str):
                raise EinopsError('bias shape should be string specifying which axes bias depends on')
            bias = ParsedExpression(bias_shape)
            _report_axes(
                set.difference(bias.identifiers, right.identifiers),
                'Bias axes {} not present in output'
            )
            _report_axes(
                set.difference(bias.identifiers, set(axes_lengths)),
                'Sizes not provided for bias axes {}',
            )

            _bias_shape = []
            for axes in right.composition:
                for axis in axes:
                    if axis in bias.identifiers:
                        _bias_shape.append(axes_lengths[axis])
                    else:
                        _bias_shape.append(1)
        else:
            _bias_shape = None
            _bias_input_size = None

        weight_bound = (3 / _fan_in) ** 0.5
        bias_bound = (1 / _fan_in) ** 0.5
        self._create_parameters(_weight_shape, weight_bound, _bias_shape, bias_bound)

        # rewrite einsum expression with single-letter latin identifiers so that each expression is
        mapping2letters = {*left.identifiers, *right.identifiers, *weight.identifiers}
        mapping2letters = {k: letter for letter, k in zip(string.ascii_lowercase, mapping2letters)}

        def write_flat(axes: list):
            return ''.join(mapping2letters[axis] for axis in axes)

        self.einsum_pattern = '{},{}->{}'.format(
            write_flat(left.flat_axes_order()),
            write_flat(weight.flat_axes_order()),
            write_flat(right.flat_axes_order()),
        )

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        """ Shape and implementations """
        raise NotImplementedError('Should be defined in framework implementations')

    def __repr__(self):
        params = repr(self.pattern)
        params += ', ' + self.weight_shape
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)
