import dataclasses
import typing

import numpy as np
import pytest

from einops import EinopsError, asnumpy, pack, unpack
from einops.tests import collect_test_backends


def pack_unpack(xs, pattern):
    x, ps = pack(xs, pattern)
    unpacked = unpack(xs, ps, pattern)
    assert len(unpacked) == len(xs)
    for a, b in zip(unpacked, xs):
        assert np.allclose(asnumpy(a), asnumpy(b))


def unpack_and_pack(x, ps, pattern: str):
    unpacked = unpack(x, ps, pattern)
    packed, ps2 = pack(unpacked, pattern=pattern)

    assert np.allclose(asnumpy(packed), asnumpy(x))
    return unpacked


def unpack_and_pack_against_numpy(x, ps, pattern: str):
    capturer_backend = CaptureException()
    capturer_numpy = CaptureException()

    with capturer_backend:
        unpacked = unpack(x, ps, pattern)
        packed, ps2 = pack(unpacked, pattern=pattern)

    with capturer_numpy:
        x_np = asnumpy(x)
        unpacked_np = unpack(x_np, ps, pattern)
        packed_np, ps3 = pack(unpacked_np, pattern=pattern)

    assert type(capturer_numpy.exception) == type(capturer_backend.exception)  # noqa E721
    if capturer_numpy.exception is not None:
        # both failed
        return
    else:
        # neither failed, check results are identical
        assert np.allclose(asnumpy(packed), asnumpy(x))
        assert np.allclose(asnumpy(packed_np), asnumpy(x))
        assert len(unpacked) == len(unpacked_np)
        for a, b in zip(unpacked, unpacked_np):
            assert np.allclose(asnumpy(a), b)


class CaptureException:
    def __enter__(self):
        self.exception = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exception = exc_val
        return True


def test_numpy_trivial(H=13, W=17):
    def rand(*shape):
        return np.random.random(shape)

    def check(a, b):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert np.all(a == b)

    r, g, b = rand(3, H, W)
    embeddings = rand(H, W, 32)

    check(
        np.stack([r, g, b], axis=2),
        pack([r, g, b], "h w *")[0],
    )
    check(
        np.stack([r, g, b], axis=1),
        pack([r, g, b], "h * w")[0],
    )
    check(
        np.stack([r, g, b], axis=0),
        pack([r, g, b], "* h w")[0],
    )

    check(
        np.concatenate([r, g, b], axis=1),
        pack([r, g, b], "h *")[0],
    )
    check(
        np.concatenate([r, g, b], axis=0),
        pack([r, g, b], "* w")[0],
    )

    i = np.index_exp[:, :, None]
    check(
        np.concatenate([r[i], g[i], b[i], embeddings], axis=2),
        pack([r, g, b, embeddings], "h w *")[0],
    )

    with pytest.raises(EinopsError):
        pack([r, g, b, embeddings], "h w nonexisting_axis *")

    pack([r, g, b], "some_name_for_H some_name_for_w1 *")

    with pytest.raises(EinopsError):
        pack([r, g, b, embeddings], "h _w *")  # no leading underscore
    with pytest.raises(EinopsError):
        pack([r, g, b, embeddings], "h_ w *")  # no trailing underscore
    with pytest.raises(EinopsError):
        pack([r, g, b, embeddings], "1h_ w *")
    with pytest.raises(EinopsError):
        pack([r, g, b, embeddings], "1 w *")
    with pytest.raises(EinopsError):
        pack([r, g, b, embeddings], "h h *")
    # capital and non-capital are different
    pack([r, g, b, embeddings], "h H *")


@dataclasses.dataclass
class UnpackTestCase:
    shape: typing.Tuple[int, ...]
    pattern: str

    def dim(self):
        return self.pattern.split().index("*")

    def selfcheck(self):
        assert self.shape[self.dim()] == 5


cases = [
    # NB: in all cases unpacked axis is of length 5.
    # that's actively used in tests below
    UnpackTestCase((5,), "*"),
    UnpackTestCase((5, 7), "* seven"),
    UnpackTestCase((7, 5), "seven *"),
    UnpackTestCase((5, 3, 4), "* three four"),
    UnpackTestCase((4, 5, 3), "four * three"),
    UnpackTestCase((3, 4, 5), "three four *"),
]


def test_pack_unpack_with_numpy():
    case: UnpackTestCase

    for case in cases:
        shape = case.shape
        pattern = case.pattern

        x = np.random.random(shape)
        # all correct, no minus 1
        unpack_and_pack(x, [[2], [1], [2]], pattern)
        # no -1, asking for wrong shapes
        with pytest.raises(BaseException):
            unpack_and_pack(x, [[2], [1], [2]], pattern + " non_existent_axis")
        with pytest.raises(BaseException):
            unpack_and_pack(x, [[2], [1], [1]], pattern)
        with pytest.raises(BaseException):
            unpack_and_pack(x, [[4], [1], [1]], pattern)
        # all correct, with -1
        unpack_and_pack(x, [[2], [1], [-1]], pattern)
        unpack_and_pack(x, [[2], [-1], [2]], pattern)
        unpack_and_pack(x, [[-1], [1], [2]], pattern)
        _, _, last = unpack_and_pack(x, [[2], [3], [-1]], pattern)
        assert last.shape[case.dim()] == 0
        # asking for more elements than available
        with pytest.raises(BaseException):
            unpack(x, [[2], [4], [-1]], pattern)
        # this one does not raise, because indexing x[2:1] just returns zero elements
        # with pytest.raises(BaseException):
        #     unpack(x, [[2], [-1], [4]], pattern)
        with pytest.raises(BaseException):
            unpack(x, [[-1], [1], [5]], pattern)

        # all correct, -1 nested
        rs = unpack_and_pack(x, [[1, 2], [1, 1], [-1, 1]], pattern)
        assert all(len(r.shape) == len(x.shape) + 1 for r in rs)
        rs = unpack_and_pack(x, [[1, 2], [1, -1], [1, 1]], pattern)
        assert all(len(r.shape) == len(x.shape) + 1 for r in rs)
        rs = unpack_and_pack(x, [[2, -1], [1, 2], [1, 1]], pattern)
        assert all(len(r.shape) == len(x.shape) + 1 for r in rs)

        # asking for more elements, -1 nested
        with pytest.raises(BaseException):
            unpack(x, [[-1, 2], [1], [5]], pattern)
        with pytest.raises(BaseException):
            unpack(x, [[2, 2], [2], [5, -1]], pattern)

        # asking for non-divisible number of elements
        with pytest.raises(BaseException):
            unpack(x, [[2, 1], [1], [3, -1]], pattern)
        with pytest.raises(BaseException):
            unpack(x, [[2, 1], [3, -1], [1]], pattern)
        with pytest.raises(BaseException):
            unpack(x, [[3, -1], [2, 1], [1]], pattern)

        # -1 takes zero
        unpack_and_pack(x, [[0], [5], [-1]], pattern)
        unpack_and_pack(x, [[0], [-1], [5]], pattern)
        unpack_and_pack(x, [[-1], [5], [0]], pattern)

        # -1 takes zero, -1
        unpack_and_pack(x, [[2, -1], [1, 5]], pattern)


def test_pack_unpack_against_numpy():
    for backend in collect_test_backends(symbolic=False, layers=False):
        print(f"test packing against numpy for {backend.framework_name}")
        check_zero_len = True

        for case in cases:
            unpack_and_pack = unpack_and_pack_against_numpy
            shape = case.shape
            pattern = case.pattern

            x = np.random.random(shape)
            x = backend.from_numpy(x)
            # all correct, no minus 1
            unpack_and_pack(x, [[2], [1], [2]], pattern)
            # no -1, asking for wrong shapes
            with pytest.raises(BaseException):
                unpack(x, [[2], [1], [1]], pattern)

            with pytest.raises(BaseException):
                unpack(x, [[4], [1], [1]], pattern)
            # all correct, with -1
            unpack_and_pack(x, [[2], [1], [-1]], pattern)
            unpack_and_pack(x, [[2], [-1], [2]], pattern)
            unpack_and_pack(x, [[-1], [1], [2]], pattern)

            # asking for more elements than available
            with pytest.raises(BaseException):
                unpack(x, [[2], [4], [-1]], pattern)
            # this one does not raise, because indexing x[2:1] just returns zero elements
            # with pytest.raises(BaseException):
            #     unpack(x, [[2], [-1], [4]], pattern)
            with pytest.raises(BaseException):
                unpack(x, [[-1], [1], [5]], pattern)

            # all correct, -1 nested
            unpack_and_pack(x, [[1, 2], [1, 1], [-1, 1]], pattern)
            unpack_and_pack(x, [[1, 2], [1, -1], [1, 1]], pattern)
            unpack_and_pack(x, [[2, -1], [1, 2], [1, 1]], pattern)

            # asking for more elements, -1 nested
            with pytest.raises(BaseException):
                unpack(x, [[-1, 2], [1], [5]], pattern)
            with pytest.raises(BaseException):
                unpack(x, [[2, 2], [2], [5, -1]], pattern)

            # asking for non-divisible number of elements
            with pytest.raises(BaseException):
                unpack(x, [[2, 1], [1], [3, -1]], pattern)
            with pytest.raises(BaseException):
                unpack(x, [[2, 1], [3, -1], [1]], pattern)
            with pytest.raises(BaseException):
                unpack(x, [[3, -1], [2, 1], [1]], pattern)

            if check_zero_len:
                # -1 takes zero
                unpack_and_pack(x, [[2], [3], [-1]], pattern)
                unpack_and_pack(x, [[0], [5], [-1]], pattern)
                unpack_and_pack(x, [[0], [-1], [5]], pattern)
                unpack_and_pack(x, [[-1], [5], [0]], pattern)

                # -1 takes zero, -1
                unpack_and_pack(x, [[2, -1], [1, 5]], pattern)


def test_pack_unpack_array_api():
    from einops import array_api as AA
    import numpy as xp

    if xp.__version__ < "2.0.0":
        pytest.skip()

    for case in cases:
        shape = case.shape
        pattern = case.pattern
        x_np = np.random.random(shape)
        x_xp = xp.from_dlpack(x_np)

        for ps in [
            [[2], [1], [2]],
            [[1], [1], [-1]],
            [[1], [1], [-1, 3]],
            [[2, 1], [1, 1, 1], [-1]],
        ]:
            x_np_split = unpack(x_np, ps, pattern)
            x_xp_split = AA.unpack(x_xp, ps, pattern)
            for a, b in zip(x_np_split, x_xp_split):
                assert np.allclose(a, AA.asnumpy(b + 0))

            x_agg_np, ps1 = pack(x_np_split, pattern)
            x_agg_xp, ps2 = AA.pack(x_xp_split, pattern)
            assert ps1 == ps2
            assert np.allclose(x_agg_np, AA.asnumpy(x_agg_xp))

        for ps in [
            [[2, 3]],
            [[1], [5]],
            [[1], [5], [-1]],
            [[1], [2, 3]],
            [[1], [5], [-1, 2]],
        ]:
            with pytest.raises(BaseException):
                unpack(x_np, ps, pattern)
