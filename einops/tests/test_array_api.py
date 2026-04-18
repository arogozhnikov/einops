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
