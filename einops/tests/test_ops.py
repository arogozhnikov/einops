import itertools

import numpy
import numpy as np
import pytest

from einops import EinopsError
from einops.einops import rearrange, reduce, repeat, _enumerate_directions
from einops.tests import collect_test_backends, is_backend_tested, FLOAT_REDUCTIONS as REDUCTIONS

imp_op_backends = collect_test_backends(symbolic=False, layers=False)
sym_op_backends = collect_test_backends(symbolic=True, layers=False)

identity_patterns = [
    "...->...",
    "a b c d e-> a b c d e",
    "a b c d e ...-> ... a b c d e",
    "a b c d e ...-> a ... b c d e",
    "... a b c d e -> ... a b c d e",
    "a ... e-> a ... e",
    "a ... -> a ... ",
    "a ... c d e -> a (...) c d e",
]

equivalent_rearrange_patterns = [
    ("a b c d e -> (a b) c d e", "a b ... -> (a b) ... "),
    ("a b c d e -> a b (c d) e", "... c d e -> ... (c d) e"),
    ("a b c d e -> a b c d e", "... -> ... "),
    ("a b c d e -> (a b c d e)", "... ->  (...)"),
    ("a b c d e -> b (c d e) a", "a b ... -> b (...) a"),
    ("a b c d e -> b (a c d) e", "a b ... e -> b (a ...) e"),
]

equivalent_reduction_patterns = [
    ("a b c d e -> ", " ... ->  "),
    ("a b c d e -> (e a)", "a ... e -> (e a)"),
    ("a b c d e -> d (a e)", " a b c d e ... -> d (a e) "),
    ("a b c d e -> (a b)", " ... c d e  -> (...) "),
]


def test_collapsed_ellipsis_errors_out():
    x = numpy.zeros([1, 1, 1, 1, 1])
    rearrange(x, "a b c d ... ->  a b c ... d")
    with pytest.raises(EinopsError):
        rearrange(x, "a b c d (...) ->  a b c ... d")

    rearrange(x, "... ->  (...)")
    with pytest.raises(EinopsError):
        rearrange(x, "(...) -> (...)")


def test_ellipsis_ops_numpy():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in identity_patterns:
        assert numpy.array_equal(x, rearrange(x, pattern)), pattern

    for pattern1, pattern2 in equivalent_rearrange_patterns:
        assert numpy.array_equal(rearrange(x, pattern1), rearrange(x, pattern2))

    for reduction in ["min", "max", "sum"]:
        for pattern1, pattern2 in equivalent_reduction_patterns:
            assert numpy.array_equal(reduce(x, pattern1, reduction=reduction), reduce(x, pattern2, reduction=reduction))

    # now just check coincidence with numpy
    all_rearrange_patterns = [*identity_patterns]
    for pattern_pairs in equivalent_rearrange_patterns:
        all_rearrange_patterns.extend(pattern_pairs)


def check_op_against_numpy(backend, numpy_input, pattern, axes_lengths, reduction="rearrange", is_symbolic=False):
    """
    Helper to test result of operation (rearrange or transpose) against numpy
    if reduction == 'rearrange', rearrange op is tested, otherwise reduce
    """

    def operation(x):
        if reduction == "rearrange":
            return rearrange(x, pattern, **axes_lengths)
        else:
            return reduce(x, pattern, reduction, **axes_lengths)

    numpy_result = operation(numpy_input)
    check_equal = numpy.array_equal
    p_none_dimension = 0.5
    if is_symbolic:
        symbol_shape = [d if numpy.random.random() >= p_none_dimension else None for d in numpy_input.shape]
        symbol = backend.create_symbol(shape=symbol_shape)
        result_symbol = operation(symbol)
        backend_result = backend.eval_symbol(result_symbol, [(symbol, numpy_input)])
    else:
        backend_result = operation(backend.from_numpy(numpy_input))
        backend_result = backend.to_numpy(backend_result)

    check_equal(numpy_result, backend_result)


def test_ellipsis_ops_imperative():
    """Checking various patterns against numpy"""
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for is_symbolic in [True, False]:
        for backend in collect_test_backends(symbolic=is_symbolic, layers=False):
            for pattern in identity_patterns + list(itertools.chain(*equivalent_rearrange_patterns)):
                check_op_against_numpy(
                    backend, x, pattern, axes_lengths={}, reduction="rearrange", is_symbolic=is_symbolic
                )

            for reduction in ["min", "max", "sum"]:
                for pattern in itertools.chain(*equivalent_reduction_patterns):
                    check_op_against_numpy(
                        backend, x, pattern, axes_lengths={}, reduction=reduction, is_symbolic=is_symbolic
                    )


def test_rearrange_array_api():
    import numpy as xp
    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in identity_patterns + list(itertools.chain(*equivalent_rearrange_patterns)):
        expected = rearrange(x, pattern)
        result = AA.rearrange(xp.from_dlpack(x), pattern)
        assert numpy.array_equal(AA.asnumpy(result + 0), expected)


def test_reduce_array_api():
    import numpy as xp
    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])
    for pattern in itertools.chain(*equivalent_reduction_patterns):
        for reduction in ["min", "max", "sum"]:
            expected = reduce(x, pattern, reduction=reduction)
            result = AA.reduce(xp.from_dlpack(x), pattern, reduction=reduction)
            assert numpy.array_equal(AA.asnumpy(np.asarray(result + 0)), expected)


def test_rearrange_consistency_numpy():
    shape = [1, 2, 3, 5, 7, 11]
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    for pattern in [
        "a b c d e f -> a b c d e f",
        "b a c d e f -> a b d e f c",
        "a b c d e f -> f e d c b a",
        "a b c d e f -> (f e) d (c b a)",
        "a b c d e f -> (f e d c b a)",
    ]:
        result = rearrange(x, pattern)
        assert len(numpy.setdiff1d(x, result)) == 0
        assert result.dtype == x.dtype

    result = rearrange(x, "a b c d e f -> a (b) (c d e) f")
    assert numpy.array_equal(x.flatten(), result.flatten())

    result = rearrange(x, "a aa aa1 a1a1 aaaa a11 -> a aa aa1 a1a1 aaaa a11")
    assert numpy.array_equal(x, result)

    result1 = rearrange(x, "a b c d e f -> f e d c b a")
    result2 = rearrange(x, "f e d c b a -> a b c d e f")
    assert numpy.array_equal(result1, result2)

    result = rearrange(rearrange(x, "a b c d e f -> (f d) c (e b) a"), "(f d) c (e b) a -> a b c d e f", b=2, d=5)
    assert numpy.array_equal(x, result)

    sizes = dict(zip("abcdef", shape))
    temp = rearrange(x, "a b c d e f -> (f d) c (e b) a", **sizes)
    result = rearrange(temp, "(f d) c (e b) a -> a b c d e f", **sizes)
    assert numpy.array_equal(x, result)

    x2 = numpy.arange(2 * 3 * 4).reshape([2, 3, 4])
    result = rearrange(x2, "a b c -> b c a")
    assert x2[1, 2, 3] == result[2, 3, 1]
    assert x2[0, 1, 2] == result[1, 2, 0]


def test_rearrange_permutations_numpy():
    # tests random permutation of axes against two independent numpy ways
    for n_axes in range(1, 10):
        input = numpy.arange(2**n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = " ".join("i" + str(axis) for axis in range(n_axes))
        right_expression = " ".join("i" + str(axis) for axis in permutation)
        expression = left_expression + " -> " + right_expression
        result = rearrange(input, expression)

        for pick in numpy.random.randint(0, 2, [10, n_axes]):
            assert input[tuple(pick)] == result[tuple(pick[permutation])]

    for n_axes in range(1, 10):
        input = numpy.arange(2**n_axes).reshape([2] * n_axes)
        permutation = numpy.random.permutation(n_axes)
        left_expression = " ".join("i" + str(axis) for axis in range(n_axes)[::-1])
        right_expression = " ".join("i" + str(axis) for axis in permutation[::-1])
        expression = left_expression + " -> " + right_expression
        result = rearrange(input, expression)
        assert result.shape == input.shape
        expected_result = numpy.zeros_like(input)
        for original_axis, result_axis in enumerate(permutation):
            expected_result |= ((input >> original_axis) & 1) << result_axis

        assert numpy.array_equal(result, expected_result)


def test_reduction_imperatives():
    for backend in imp_op_backends:
        print("Reduction tests for ", backend.framework_name)
        for reduction in REDUCTIONS:
            # slight redundancy for simpler order - numpy version is evaluated multiple times
            input = numpy.arange(2 * 3 * 4 * 5 * 6, dtype="int64").reshape([2, 3, 4, 5, 6])
            if reduction in ["mean", "prod"]:
                input = input / input.astype("float64").mean()
            test_cases = [
                ["a b c d e -> ", {}, getattr(input, reduction)()],
                ["a ... -> ", {}, getattr(input, reduction)()],
                ["(a1 a2) ... (e1 e2) -> ", dict(a1=1, e2=2), getattr(input, reduction)()],
                [
                    "a b c d e -> (e c) a",
                    {},
                    getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
                ],
                [
                    "a ... c d e -> (e c) a",
                    {},
                    getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
                ],
                [
                    "a b c d e ... -> (e c) a",
                    {},
                    getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
                ],
                ["a b c d e -> (e c a)", {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1])],
                ["(a a2) ... -> (a2 a) ...", dict(a2=1), input],
            ]
            for pattern, axes_lengths, expected_result in test_cases:
                result = reduce(backend.from_numpy(input.copy()), pattern, reduction=reduction, **axes_lengths)
                result = backend.to_numpy(result)
                assert numpy.allclose(result, expected_result), f"Failed at {pattern}"


def test_reduction_symbolic():
    for backend in sym_op_backends:
        print("Reduction tests for ", backend.framework_name)
        for reduction in REDUCTIONS:
            input = numpy.arange(2 * 3 * 4 * 5 * 6, dtype="int64").reshape([2, 3, 4, 5, 6])
            input = input / input.astype("float64").mean()
            # slight redundancy for simpler order - numpy version is evaluated multiple times
            test_cases = [
                ["a b c d e -> ", {}, getattr(input, reduction)()],
                ["a ... -> ", {}, getattr(input, reduction)()],
                ["(a a2) ... (e e2) -> ", dict(a2=1, e2=1), getattr(input, reduction)()],
                [
                    "a b c d e -> (e c) a",
                    {},
                    getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
                ],
                [
                    "a ... c d e -> (e c) a",
                    {},
                    getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
                ],
                [
                    "a b c d e ... -> (e c) a",
                    {},
                    getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1, 2]),
                ],
                ["a b c d e -> (e c a)", {}, getattr(input, reduction)(axis=(1, 3)).transpose(2, 1, 0).reshape([-1])],
                ["(a a2) ... -> (a2 a) ...", dict(a2=1), input],
            ]
            for pattern, axes_lengths, expected_numpy_result in test_cases:
                shapes = [input.shape, [None for _ in input.shape]]
                for shape in shapes:
                    sym = backend.create_symbol(shape)
                    result_sym = reduce(sym, pattern, reduction=reduction, **axes_lengths)
                    result = backend.eval_symbol(result_sym, [(sym, input)])
                    assert numpy.allclose(result, expected_numpy_result)

                if True:
                    shape = []
                    _axes_lengths = {**axes_lengths}
                    for axis, length in zip("abcde", input.shape):
                        # filling as much as possible with Nones
                        if axis in pattern:
                            shape.append(None)
                            _axes_lengths[axis] = length
                        else:
                            shape.append(length)
                    sym = backend.create_symbol(shape)
                    result_sym = reduce(sym, pattern, reduction=reduction, **_axes_lengths)
                    result = backend.eval_symbol(result_sym, [(sym, input)])
                    assert numpy.allclose(result, expected_numpy_result)


def test_reduction_stress_imperatives():
    for backend in imp_op_backends:
        print("Stress-testing reduction for ", backend.framework_name)
        for reduction in REDUCTIONS + ("rearrange",):
            dtype = "int64"
            coincide = numpy.array_equal
            if reduction in ["mean", "prod"]:
                dtype = "float64"
                coincide = numpy.allclose
            max_dim = 11
            if "oneflow" in backend.framework_name:
                max_dim = 7
            if "paddle" in backend.framework_name:
                max_dim = 9
            for n_axes in range(max_dim):
                shape = numpy.random.randint(2, 4, size=n_axes)
                permutation = numpy.random.permutation(n_axes)
                skipped = 0 if reduction == "rearrange" else numpy.random.randint(n_axes + 1)
                left = " ".join("x" + str(i) for i in range(n_axes))
                right = " ".join("x" + str(i) for i in permutation[skipped:])
                pattern = left + "->" + right
                x = numpy.arange(1, 1 + numpy.prod(shape), dtype=dtype).reshape(shape)
                if reduction == "prod":
                    x /= x.mean()  # to avoid overflows
                result1 = reduce(x, pattern, reduction=reduction)
                result2 = x.transpose(permutation)
                if skipped > 0:
                    result2 = getattr(result2, reduction)(axis=tuple(range(skipped)))
                assert coincide(result1, result2)
                check_op_against_numpy(backend, x, pattern, reduction=reduction, axes_lengths={}, is_symbolic=False)


def test_reduction_with_callable_imperatives():
    x_numpy = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6]).astype("float32")
    x_numpy /= x_numpy.max()

    def logsumexp_torch(x, tuple_of_axes):
        return x.logsumexp(tuple_of_axes)

    def logsumexp_tf(x, tuple_of_axes):
        import tensorflow as tf

        return tf.reduce_logsumexp(x, tuple_of_axes)

    def logsumexp_keras(x, tuple_of_axes):
        import tensorflow.keras.backend as k

        return k.logsumexp(x, tuple_of_axes)

    def logsumexp_numpy(x, tuple_of_axes):
        # very naive logsumexp to compare to
        minused = x.max(tuple_of_axes)
        y = x - x.max(tuple_of_axes, keepdims=True)
        y = numpy.exp(y)
        y = numpy.sum(y, axis=tuple_of_axes)
        return numpy.log(y) + minused

    from einops._backends import TorchBackend, TensorflowBackend, TFKerasBackend, NumpyBackend

    backend2callback = {
        TorchBackend.framework_name: logsumexp_torch,
        TensorflowBackend.framework_name: logsumexp_tf,
        TFKerasBackend.framework_name: logsumexp_keras,
        NumpyBackend.framework_name: logsumexp_numpy,
    }

    for backend in imp_op_backends:
        if backend.framework_name not in backend2callback:
            continue

        backend_callback = backend2callback[backend.framework_name]

        x_backend = backend.from_numpy(x_numpy)
        for pattern1, pattern2 in equivalent_reduction_patterns:
            print("Test reduction with callable for ", backend.framework_name, pattern1, pattern2)
            output_numpy = reduce(x_numpy, pattern1, reduction=logsumexp_numpy)
            output_backend = reduce(x_backend, pattern1, reduction=backend_callback)
            assert numpy.allclose(
                output_numpy,
                backend.to_numpy(output_backend),
            )


def test_enumerating_directions():
    for backend in imp_op_backends:
        print("testing directions for", backend.framework_name)
        for shape in [[], [1], [1, 1, 1], [2, 3, 5, 7]]:
            x = numpy.arange(numpy.prod(shape)).reshape(shape)
            axes1 = _enumerate_directions(x)
            axes2 = _enumerate_directions(backend.from_numpy(x))
            assert len(axes1) == len(axes2) == len(shape)
            for ax1, ax2 in zip(axes1, axes2):
                ax2 = backend.to_numpy(ax2)
                assert ax1.shape == ax2.shape
                assert numpy.allclose(ax1, ax2)


def test_concatenations_and_stacking():
    for backend in imp_op_backends:
        print("testing shapes for ", backend.framework_name)
        for n_arrays in [1, 2, 5]:
            shapes = [[], [1], [1, 1], [2, 3, 5, 7], [1] * 6]
            for shape in shapes:
                arrays1 = [numpy.arange(i, i + numpy.prod(shape)).reshape(shape) for i in range(n_arrays)]
                arrays2 = [backend.from_numpy(array) for array in arrays1]
                result0 = numpy.asarray(arrays1)
                result1 = rearrange(arrays1, "...->...")
                result2 = rearrange(arrays2, "...->...")
                assert numpy.array_equal(result0, result1)
                assert numpy.array_equal(result1, backend.to_numpy(result2))

                result1 = rearrange(arrays1, "b ... -> ... b")
                result2 = rearrange(arrays2, "b ... -> ... b")
                assert numpy.array_equal(result1, backend.to_numpy(result2))


def test_gradients_imperatives():
    # lazy - just checking reductions
    for reduction in REDUCTIONS:
        if reduction in ("any", "all"):
            continue  # non-differentiable ops
        x = numpy.arange(1, 1 + 2 * 3 * 4).reshape([2, 3, 4]).astype("float32")
        results = {}
        for backend in imp_op_backends:
            y0 = backend.from_numpy(x)
            if not hasattr(y0, "grad"):
                continue

            y1 = reduce(y0, "a b c -> c a", reduction=reduction)
            y2 = reduce(y1, "c a -> a c", reduction=reduction)
            y3 = reduce(y2, "a (c1 c2) -> a", reduction=reduction, c1=2)
            y4 = reduce(y3, "... -> ", reduction=reduction)

            y4.backward()
            grad = backend.to_numpy(y0.grad)
            results[backend.framework_name] = grad

        print("comparing gradients for", results.keys())
        for name1, grad1 in results.items():
            for name2, grad2 in results.items():
                assert numpy.allclose(grad1, grad2), [name1, name2, "provided different gradients"]


def test_tiling_imperatives():
    for backend in imp_op_backends:
        print("Tiling tests for ", backend.framework_name)
        input = numpy.arange(2 * 3 * 5, dtype="int64").reshape([2, 1, 3, 1, 5])
        test_cases = [
            (1, 1, 1, 1, 1),
            (1, 2, 1, 3, 1),
            (3, 1, 1, 4, 1),
        ]
        for repeats in test_cases:
            expected = numpy.tile(input, repeats)
            converted = backend.from_numpy(input)
            repeated = backend.tile(converted, repeats)
            result = backend.to_numpy(repeated)
            assert numpy.array_equal(result, expected)


def test_tiling_symbolic():
    for backend in sym_op_backends:
        print("Tiling tests for ", backend.framework_name)
        input = numpy.arange(2 * 3 * 5, dtype="int64").reshape([2, 1, 3, 1, 5])
        test_cases = [
            (1, 1, 1, 1, 1),
            (1, 2, 1, 3, 1),
            (3, 1, 1, 4, 1),
        ]
        for repeats in test_cases:
            expected = numpy.tile(input, repeats)
            sym = backend.create_symbol(input.shape)
            result = backend.eval_symbol(backend.tile(sym, repeats), [[sym, input]])
            assert numpy.array_equal(result, expected)

            sym = backend.create_symbol([None] * len(input.shape))
            result = backend.eval_symbol(backend.tile(sym, repeats), [[sym, input]])
            assert numpy.array_equal(result, expected)


repeat_test_cases = [
    # all assume that input has shape [2, 3, 5]
    ("a b c -> c a b", dict()),
    ("a b c -> (c copy a b)", dict(copy=2, a=2, b=3, c=5)),
    ("a b c -> (a copy) b c ", dict(copy=1)),
    ("a b c -> (c a) (copy1 b copy2)", dict(a=2, copy1=1, copy2=2)),
    ("a ...  -> a ... copy", dict(copy=4)),
    ("... c -> ... (copy1 c copy2)", dict(copy1=1, copy2=2)),
    ("...  -> ... ", dict()),
    (" ...  -> copy1 ... copy2 ", dict(copy1=2, copy2=3)),
    ("a b c  -> copy1 a copy2 b c () ", dict(copy1=2, copy2=1)),
]


def check_reversion(x, repeat_pattern, **sizes):
    """Checks repeat pattern by running reduction"""
    left, right = repeat_pattern.split("->")
    reduce_pattern = right + "->" + left
    repeated = repeat(x, repeat_pattern, **sizes)
    reduced_min = reduce(repeated, reduce_pattern, reduction="min", **sizes)
    reduced_max = reduce(repeated, reduce_pattern, reduction="max", **sizes)
    assert numpy.array_equal(x, reduced_min)
    assert numpy.array_equal(x, reduced_max)


def test_repeat_numpy():
    # check repeat vs reduce. Repeat works ok if reverse reduction with min and max work well
    x = numpy.arange(2 * 3 * 5).reshape([2, 3, 5])
    x1 = repeat(x, "a b c -> copy a b c ", copy=1)
    assert numpy.array_equal(x[None], x1)
    for pattern, axis_dimensions in repeat_test_cases:
        check_reversion(x, pattern, **axis_dimensions)


def test_repeat_imperatives():
    x = numpy.arange(2 * 3 * 5).reshape([2, 3, 5])
    for backend in imp_op_backends:
        print("Repeat tests for ", backend.framework_name)

        for pattern, axis_dimensions in repeat_test_cases:
            expected = repeat(x, pattern, **axis_dimensions)
            converted = backend.from_numpy(x)
            repeated = repeat(converted, pattern, **axis_dimensions)
            result = backend.to_numpy(repeated)
            assert numpy.array_equal(result, expected)


def test_repeat_symbolic():
    x = numpy.arange(2 * 3 * 5).reshape([2, 3, 5])

    for backend in sym_op_backends:
        print("Repeat tests for ", backend.framework_name)

        for pattern, axis_dimensions in repeat_test_cases:
            expected = repeat(x, pattern, **axis_dimensions)

            sym = backend.create_symbol(x.shape)
            result = backend.eval_symbol(repeat(sym, pattern, **axis_dimensions), [[sym, x]])
            assert numpy.array_equal(result, expected)


def test_repeat_array_api():
    import numpy as xp
    from einops import array_api as AA

    if xp.__version__ < "2.0.0":
        pytest.skip()

    x = numpy.arange(2 * 3 * 5).reshape([2, 3, 5])

    for pattern, axis_dimensions in repeat_test_cases:
        expected = repeat(x, pattern, **axis_dimensions)

        result = AA.repeat(xp.from_dlpack(x), pattern, **axis_dimensions)
        assert numpy.array_equal(AA.asnumpy(result + 0), expected)


test_cases_repeat_anonymous = [
    # all assume that input has shape [1, 2, 4, 6]
    ("a b c d -> c a d b", dict()),
    ("a b c d -> (c 2 d a b)", dict(a=1, c=4, d=6)),
    ("1 b c d -> (d copy 1) 3 b c ", dict(copy=3)),
    ("1 ...  -> 3 ... ", dict()),
    ("() ... d -> 1 (copy1 d copy2) ... ", dict(copy1=2, copy2=3)),
    ("1 b c d -> (1 1) (1 b) 2 c 3 d (1 1)", dict()),
]


def test_anonymous_axes():
    x = numpy.arange(1 * 2 * 4 * 6).reshape([1, 2, 4, 6])
    for pattern, axis_dimensions in test_cases_repeat_anonymous:
        check_reversion(x, pattern, **axis_dimensions)


def test_list_inputs():
    x = numpy.arange(2 * 3 * 4 * 5 * 6).reshape([2, 3, 4, 5, 6])

    assert numpy.array_equal(
        rearrange(list(x), "... -> (...)"),
        rearrange(x, "... -> (...)"),
    )
    assert numpy.array_equal(
        reduce(list(x), "a ... e -> (...)", "min"),
        reduce(x, "a ... e -> (...)", "min"),
    )
    assert numpy.array_equal(
        repeat(list(x), "...  -> b (...)", b=3),
        repeat(x, "...  -> b (...)", b=3),
    )


def test_torch_compile_with_dynamic_shape():
    if not is_backend_tested("torch"):
        pytest.skip()
    import torch

    # somewhat reasonable debug messages
    torch._dynamo.config.verbose = True

    def func1(x):
        # test contains ellipsis
        a, b, c, *other = x.shape
        x = rearrange(x, "(a a2) b c ... -> b (c a2) (a ...)", a2=2)
        # test contains passing expression as axis length
        x = reduce(x, "b ca2 A -> b A", "sum", ca2=c * 2)
        return x

    # seems can't test static and dynamic in the same test run.
    # func1_compiled_static = torch.compile(func1, dynamic=False, fullgraph=True, backend='aot_eager')
    func1_compiled_dynamic = torch.compile(func1, dynamic=True, fullgraph=True, backend="aot_eager")

    x = torch.randn(size=[4, 5, 6, 3])
    assert torch.equal(func1_compiled_dynamic(x), func1(x))
    # check with input of different dimensionality, and with all shape elements changed
    x = torch.randn(size=[6, 3, 4, 2, 3])
    assert torch.equal(func1_compiled_dynamic(x), func1(x))


def bit_count(x):
    return sum((x >> i) & 1 for i in range(20))


def test_reduction_imperatives_booleans():
    """Checks that any/all reduction works in all frameworks"""
    x_np = numpy.asarray([(bit_count(x) % 2) == 0 for x in range(2**6)]).reshape([2] * 6)
    for backend in imp_op_backends:
        print("Reduction any/all tests for ", backend.framework_name)

        for axis in range(6):
            expected_result_any = numpy.any(x_np, axis=axis, keepdims=True)
            expected_result_all = numpy.all(x_np, axis=axis, keepdims=True)
            assert not numpy.array_equal(expected_result_any, expected_result_all)

            axes = list("abcdef")
            axes_in = list(axes)
            axes_out = list(axes)
            axes_out[axis] = "1"
            pattern = (" ".join(axes_in)) + " -> " + (" ".join(axes_out))

            res_any = reduce(backend.from_numpy(x_np), pattern, reduction="any")
            res_all = reduce(backend.from_numpy(x_np), pattern, reduction="all")

            assert numpy.array_equal(expected_result_any, backend.to_numpy(res_any))
            assert numpy.array_equal(expected_result_all, backend.to_numpy(res_all))

        # expected result: any/all
        expected_result_any = numpy.any(x_np, axis=(0, 1), keepdims=True)
        expected_result_all = numpy.all(x_np, axis=(0, 1), keepdims=True)
        pattern = "a b ... -> 1 1 ..."
        res_any = reduce(backend.from_numpy(x_np), pattern, reduction="any")
        res_all = reduce(backend.from_numpy(x_np), pattern, reduction="all")
        assert numpy.array_equal(expected_result_any, backend.to_numpy(res_any))
        assert numpy.array_equal(expected_result_all, backend.to_numpy(res_all))
