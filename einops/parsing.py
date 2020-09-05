from einops import EinopsError
import keyword
import warnings
from typing import List

_ellipsis = '…'  # NB, this is a single unicode symbol. String is used as it is not a list, but can be iterated


class AnonymousAxis(object):
    """Important thing: all instances of this class are not equal to each other """

    def __init__(self, value: str):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise EinopsError('No need to create anonymous axis of length 1. Report this as an issue')
            else:
                raise EinopsError('Anonymous axis should have positive length, not {}'.format(self.value))

    def __repr__(self):
        return "{}-axis".format(str(self.value))


class ParsedExpression:
    """
    non-mutable structure that contains information about one side of expression (e.g. 'b c (h w)')
    and keeps some information important for downstream
    """
    def __init__(self, expression):
        self.has_ellipsis = False
        self.has_ellipsis_parenthesized = None
        self.identifiers = set()
        # that's axes like 2, 3 or 5. Axes with size 1 are exceptional and replaced with empty composition
        self.has_non_unitary_anonymous_axes = False
        # composition keeps structure of composite axes, see how different corner cases are handled in tests
        self.composition = []
        if '.' in expression:
            if '...' not in expression:
                raise EinopsError('Expression may contain dots only inside ellipsis (...)')
            if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
                raise EinopsError(
                    'Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True

        bracket_group = None

        def add_axis_name(x):
            if x is not None:
                if x in self.identifiers:
                    raise EinopsError('Indexing expression contains duplicate dimension "{}"'.format(x))
                if x == _ellipsis:
                    self.identifiers.add(_ellipsis)
                    if bracket_group is None:
                        self.composition.append(_ellipsis)
                        self.has_ellipsis_parenthesized = False
                    else:
                        bracket_group.append(_ellipsis)
                        self.has_ellipsis_parenthesized = True
                else:
                    is_number = str.isdecimal(x)
                    if is_number and int(x) == 1:
                        # handling the case of anonymous axis of length 1
                        if bracket_group is None:
                            self.composition.append([])
                        else:
                            pass  # no need to think about 1s inside parenthesis
                        return
                    is_axis_name, reason = self.check_axis_name(x, return_reason=True)
                    if not (is_number or is_axis_name):
                        raise EinopsError('Invalid axis identifier: {}\n{}'.format(x, reason))
                    if is_number:
                        x = AnonymousAxis(x)
                    self.identifiers.add(x)
                    if is_number:
                        self.has_non_unitary_anonymous_axes = True
                    if bracket_group is None:
                        self.composition.append([x])
                    else:
                        bracket_group.append(x)

        current_identifier = None
        for char in expression:
            if char in '() ':
                add_axis_name(current_identifier)
                current_identifier = None
                if char == '(':
                    if bracket_group is not None:
                        raise EinopsError("Axis composition is one-level (brackets inside brackets not allowed)")
                    bracket_group = []
                elif char == ')':
                    if bracket_group is None:
                        raise EinopsError('Brackets are not balanced')
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ['_', _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise EinopsError("Unknown character '{}'".format(char))

        if bracket_group is not None:
            raise EinopsError('Imbalanced parentheses in expression: "{}"'.format(expression))
        add_axis_name(current_identifier)

    def flat_axes_order(self) -> List:
        result = []
        for composed_axis in self.composition:
            assert isinstance(composed_axis, list), 'does not work with ellipsis'
            for axis in composed_axis:
                result.append(axis)
        return result

    def has_composed_axes(self) -> bool:
        # this will ignore 1 inside brackets
        for axes in self.composition:
            if isinstance(axes, list) and len(axes) > 1:
                return True
        return False

    @staticmethod
    def check_axis_name(name: str, return_reason=False):
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        if not str.isidentifier(name):
            result = False, 'not a valid python identifier'
        elif name[0] == '_' or name[-1] == '_':
            result = False, 'axis name should should not start or end with underscore'
        else:
            if keyword.iskeyword(name):
                warnings.warn("It is discouraged to use axes names that are keywords: {}".format(name), RuntimeWarning)
            if name in ['axis']:
                warnings.warn("It is discouraged to use 'axis' as an axis name "
                              "and will raise an error in future", FutureWarning)
            result = True, None
        if return_reason:
            return result
        else:
            return result[0]
