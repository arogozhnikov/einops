import itertools

import numpy as np
import pytest

from einops import rearrange, reduce, repeat

from .test_ops import equivalent_rearrange_patterns, equivalent_reduction_patterns, identity_patterns, repeat_test_cases


def test_rearrange_array_api():
    import numpy as xp

    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    x = np.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in identity_patterns + list(itertools.chain(*equivalent_rearrange_patterns)):
        expected = rearrange(x, pattern)
        result = AA.rearrange(xp.from_dlpack(x), pattern)
        assert np.array_equal(AA.asnumpy(result + 0), expected)


def test_reduce_array_api():
    import numpy as xp

    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    x = np.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in itertools.chain(*equivalent_reduction_patterns):
        for reduction in ["min", "max", "sum"]:
            expected = reduce(x, pattern, reduction=reduction)
            result = AA.reduce(xp.from_dlpack(x), pattern, reduction=reduction)
            assert np.array_equal(AA.asnumpy(np.asarray(result + 0)), expected)


def test_reductions_any_all_for_array_api():
    import numpy as xp

    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    rng = np.random.default_rng(0)
    x = rng.random([2, 3, 4, 1, 2]) > 0.5
    for pattern in itertools.chain(*equivalent_reduction_patterns):
        for reduction in ["any", "all"]:
            expected = reduce(x, pattern, reduction=reduction)
            result = AA.reduce(xp.from_dlpack(x), pattern, reduction=reduction)
            assert np.array_equal(AA.asnumpy(np.asarray(result)), expected)


def test_repeat_array_api():
    import numpy as xp

    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    x = np.arange(2 * 3 * 5).reshape([2, 3, 5])

    for pattern, axis_dimensions in repeat_test_cases:
        expected = repeat(x, pattern, **axis_dimensions)

        result = AA.repeat(xp.from_dlpack(x), pattern, **axis_dimensions)
        assert np.array_equal(AA.asnumpy(result + 0), expected)


@pytest.mark.parametrize(
    ("shapes", "pattern", "packed_shape", "packed_shapes"),
    [
        ([(0, 3), (0, 2)], "b *", (0, 5), [(3,), (2,)]),
        ([(3, 0), (2, 0)], "* b", (5, 0), [(3,), (2,)]),
        ([(2, 0, 3), (2, 0, 2)], "a b *", (2, 0, 5), [(3,), (2,)]),
        ([(2, 3, 0), (2, 2, 0)], "a * b", (2, 5, 0), [(3,), (2,)]),
        ([(0, 2, 3), (0, 4)], "b *", (0, 10), [(2, 3), (4,)]),
        ([(0,), (0, 2)], "b *", (0, 3), [(), (2,)]),
        ([(0, 0), (0, 3)], "b *", (0, 3), [(0,), (3,)]),
        ([(2, 0), (2, 3)], "b *", (2, 3), [(0,), (3,)]),
        ([(2, 2, 3), (2, 4)], "b *", (2, 10), [(2, 3), (4,)]),
    ],
)
def test_pack_shared_zero_axes_array_api(shapes, pattern, packed_shape, packed_shapes):
    from einops import array_api as AA

    if np.__version__ < "2.0.0":
        pytest.skip()

    arrays = [np.zeros(shape) for shape in shapes]
    packed, ps = AA.pack(arrays, pattern)
    assert packed.shape == packed_shape
    assert ps == packed_shapes
    restored = AA.unpack(packed, ps, pattern)
    for tensor, array in zip(restored, arrays, strict=True):
        np.testing.assert_array_equal(tensor, array)
