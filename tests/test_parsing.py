from nose.tools import assert_raises

from einops import EinopsError
from einops.parsing import ParsedExpression, AnonymousAxis, _ellipsis

__author__ = 'Alex Rogozhnikov'


class AnonymousAxisPlaceholder:
    def __init__(self, value: int):
        self.value = value
        assert isinstance(self.value, int)

    def __eq__(self, other):
        return isinstance(other, AnonymousAxis) and self.value == other.value


def test_anonymous_axes():
    a, b = AnonymousAxis('2'), AnonymousAxis('2')
    assert a != b
    c, d = AnonymousAxisPlaceholder(2), AnonymousAxisPlaceholder(3)
    assert a == c and b == c
    assert a != d and b != d
    assert [a, 2, b] == [c, 2, c]


def test_elementary_axis_name():
    for name in ['a', 'b', 'h', 'dx', 'h1', 'zz', 'i9123', 'somelongname',
                 'Alex', 'camelCase', 'u_n_d_e_r_score', 'unreasonablyLongAxisName']:
        assert ParsedExpression.check_axis_name(name)

    for name in ['', '2b', '12', '_startWithUnderscore', 'endWithUnderscore_', '_', '...', _ellipsis]:
        assert not ParsedExpression.check_axis_name(name)


def test_invalid_expressions():
    # double ellipsis should raise an error
    ParsedExpression('... a b c d')
    with assert_raises(EinopsError):
        ParsedExpression('... a b c d ...')
    with assert_raises(EinopsError):
        ParsedExpression('... a b c (d ...)')
    with assert_raises(EinopsError):
        ParsedExpression('(... a) b c (d ...)')

    # double/missing/enclosed parenthesis
    ParsedExpression('(a) b c (d ...)')
    with assert_raises(EinopsError):
        ParsedExpression('(a)) b c (d ...)')
    with assert_raises(EinopsError):
        ParsedExpression('(a b c (d ...)')
    with assert_raises(EinopsError):
        ParsedExpression('(a) (()) b c (d ...)')
    with assert_raises(EinopsError):
        ParsedExpression('(a) ((b c) (d ...))')

    # invalid identifiers
    ParsedExpression('camelCase under_scored cApiTaLs ß ...')
    with assert_raises(EinopsError):
        ParsedExpression('1a')
    with assert_raises(EinopsError):
        ParsedExpression('_pre')
    with assert_raises(EinopsError):
        ParsedExpression('...pre')
    with assert_raises(EinopsError):
        ParsedExpression('pre...')


def test_parse_expression():
    parsed = ParsedExpression('a1  b1   c1    d1')
    assert parsed.identifiers == {'a1', 'b1', 'c1', 'd1'}
    assert parsed.composition == [['a1'], ['b1'], ['c1'], ['d1']]
    assert not parsed.has_non_unitary_anonymous_axes
    assert not parsed.has_ellipsis

    parsed = ParsedExpression('() () () ()')
    assert parsed.identifiers == set()
    assert parsed.composition == [[], [], [], []]
    assert not parsed.has_non_unitary_anonymous_axes
    assert not parsed.has_ellipsis

    parsed = ParsedExpression('1 1 1 ()')
    assert parsed.identifiers == set()
    assert parsed.composition == [[], [], [], []]
    assert not parsed.has_non_unitary_anonymous_axes
    assert not parsed.has_ellipsis

    aap = AnonymousAxisPlaceholder

    parsed = ParsedExpression('5 (3 4)')
    assert len(parsed.identifiers) == 3 and {i.value for i in parsed.identifiers} == {3, 4, 5}
    assert parsed.composition == [[aap(5)], [aap(3), aap(4)]]
    assert parsed.has_non_unitary_anonymous_axes
    assert not parsed.has_ellipsis

    parsed = ParsedExpression('5 1 (1 4) 1')
    assert len(parsed.identifiers) == 2 and {i.value for i in parsed.identifiers} == {4, 5}
    assert parsed.composition == [[aap(5)], [], [aap(4)], []]

    parsed = ParsedExpression('name1 ... a1 12 (name2 14)')
    assert len(parsed.identifiers) == 6
    assert parsed.identifiers.difference({'name1', _ellipsis, 'a1', 'name2'}).__len__() == 2
    assert parsed.composition == [['name1'], _ellipsis, ['a1'], [aap(12)], ['name2', aap(14)]]
    assert parsed.has_non_unitary_anonymous_axes
    assert parsed.has_ellipsis
    assert not parsed.has_ellipsis_parenthesized

    parsed = ParsedExpression('(name1 ... a1 12) name2 14')
    assert len(parsed.identifiers) == 6
    assert parsed.identifiers.difference({'name1', _ellipsis, 'a1', 'name2'}).__len__() == 2
    assert parsed.composition == [['name1', _ellipsis, 'a1', aap(12)], ['name2'], [aap(14)]]
    assert parsed.has_non_unitary_anonymous_axes
    assert parsed.has_ellipsis
    assert parsed.has_ellipsis_parenthesized
