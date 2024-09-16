import numpy
import pytest

from einops import rearrange, parse_shape, reduce
from einops.tests import is_backend_tested
from einops.tests.test_ops import imp_op_backends


def test_rearrange_examples():
    def test1(x):
        # transpose
        y = rearrange(x, "b c h w -> b h w c")
        assert tuple(y.shape) == (10, 30, 40, 20)
        return y

    def test2(x):
        # view / reshape
        y = rearrange(x, "b c h w -> b (c h w)")
        assert tuple(y.shape) == (10, 20 * 30 * 40)
        return y

    def test3(x):
        # depth-to-space
        y = rearrange(x, "b (c h1 w1) h w -> b c (h h1) (w w1)", h1=2, w1=2)
        assert tuple(y.shape) == (10, 5, 30 * 2, 40 * 2)
        return y

    def test4(x):
        # space-to-depth
        y = rearrange(x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", h1=2, w1=2)
        assert tuple(y.shape) == (10, 20 * 4, 30 // 2, 40 // 2)
        return y

    def test5(x):
        # simple transposition
        y = rearrange(x, "b1 sound b2 letter -> b1 b2 sound letter")
        assert tuple(y.shape) == (10, 30, 20, 40)
        return y

    def test6(x):
        # parsing parameters
        t = rearrange(x, "b c h w -> (b h w) c")
        t = t[:, ::2]  # replacement for dot-product, just changes size of second axis
        assert tuple(t.shape) == (10 * 30 * 40, 10)

        y = rearrange(t, "(b h w) c2 -> b c2 h w", **parse_shape(x, "b _ h w"))
        assert tuple(y.shape) == (10, 10, 30, 40)
        return y

    def test7(x):
        # split of embedding into groups
        y1, y2 = rearrange(x, "b (c g) h w -> g b c h w", g=2)
        assert tuple(y1.shape) == (10, 10, 30, 40)
        assert tuple(y2.shape) == (10, 10, 30, 40)
        return y1 + y2  # only one tensor is expected in output

    def test8(x):
        # max-pooling
        y = reduce(x, "b c (h h1) (w w1) -> b c h w", reduction="max", h1=2, w1=2)
        assert tuple(y.shape) == (10, 20, 30 // 2, 40 // 2)
        return y

    def test9(x):
        # squeeze - unsqueeze
        y = reduce(x, "b c h w -> b c () ()", reduction="max")
        assert tuple(y.shape) == (10, 20, 1, 1)
        y = rearrange(y, "b c () () -> c b")
        assert tuple(y.shape) == (20, 10)
        return y

    def test10(x):
        # stack
        tensors = list(x + 0)  # 0 is needed https://github.com/tensorflow/tensorflow/issues/23185
        tensors = rearrange(tensors, "b c h w -> b h w c")
        assert tuple(tensors.shape) == (10, 30, 40, 20)
        return tensors

    def test11(x):
        # concatenate
        tensors = list(x + 0)  # 0 is needed https://github.com/tensorflow/tensorflow/issues/23185
        tensors = rearrange(tensors, "b c h w -> h (b w) c")
        assert tuple(tensors.shape) == (30, 10 * 40, 20)
        return tensors

    def shufflenet(x, convolve, c1, c2):
        # shufflenet reordering example
        x = convolve(x)
        x = rearrange(x, "b (c1 c2) h w-> b (c2 c1) h w", c1=c1, c2=c2)
        x = convolve(x)
        return x

    def convolve_strided_1d(x, stride, usual_convolution):
        x = rearrange(x, "b c t1 t2 -> b c (t1 t2)")  # reduce dimensionality
        x = rearrange(x, "b c (t stride) -> (stride b) c t", stride=stride)
        x = usual_convolution(x)
        x = rearrange(x, "(stride b) c t -> b c (t stride)", stride=stride)
        return x

    def convolve_strided_2d(x, h_stride, w_stride, usual_convolution):
        x = rearrange(x, "b c (h hs) (w ws) -> (hs ws b) c h w", hs=h_stride, ws=w_stride)
        x = usual_convolution(x)
        x = rearrange(x, "(hs ws b) c h w -> b c (h hs) (w ws)", hs=h_stride, ws=w_stride)
        return x

    def unet_like_1d(x, usual_convolution):
        # u-net like steps for increasing / reducing dimensionality
        x = rearrange(x, "b c t1 t2 -> b c (t1 t2)")  # reduce dimensionality
        y = rearrange(x, "b c (t dt) -> b (dt c) t", dt=2)
        y = usual_convolution(y)
        x = x + rearrange(y, "b (dt c) t -> b c (t dt)", dt=2)
        return x

    # mock for convolution (works for all backends)
    def convolve_mock(x):
        return x

    tests = [
        test1,
        test2,
        test3,
        test4,
        test5,
        test6,
        test7,
        test8,
        test9,
        test10,
        test11,
        lambda x: shufflenet(x, convolve=convolve_mock, c1=4, c2=5),
        lambda x: convolve_strided_1d(x, stride=2, usual_convolution=convolve_mock),
        lambda x: convolve_strided_2d(x, h_stride=2, w_stride=2, usual_convolution=convolve_mock),
        lambda x: unet_like_1d(x, usual_convolution=convolve_mock),
    ]

    for backend in imp_op_backends:
        print("testing source_examples for ", backend.framework_name)
        for test in tests:
            x = numpy.arange(10 * 20 * 30 * 40).reshape([10, 20, 30, 40])
            result1 = test(x)
            result2 = backend.to_numpy(test(backend.from_numpy(x)))
            assert numpy.array_equal(result1, result2)

            # now with strides
            x = numpy.arange(10 * 2 * 20 * 3 * 30 * 1 * 40).reshape([10 * 2, 20 * 3, 30 * 1, 40 * 1])
            # known torch bug - torch doesn't support negative steps
            last_step = -1 if (backend.framework_name != "torch" and backend.framework_name != "oneflow") else 1
            indexing_expression = numpy.index_exp[::2, ::3, ::1, ::last_step]
            result1 = test(x[indexing_expression])
            result2 = backend.to_numpy(test(backend.from_numpy(x)[indexing_expression]))
            assert numpy.array_equal(result1, result2)


def tensor_train_example_numpy():
    # kept here just for a collection, only tested for numpy
    # https://arxiv.org/pdf/1509.06569.pdf, (5)
    x = numpy.ones([3, 4, 5, 6])
    rank = 4
    if numpy.__version__ < "1.15.0":
        # numpy.einsum fails here, skip test
        return
    # creating appropriate Gs
    Gs = [numpy.ones([d, d, rank, rank]) for d in x.shape]
    Gs[0] = Gs[0][:, :, :1, :]
    Gs[-1] = Gs[-1][:, :, :, :1]

    # einsum way
    y = x.reshape((1,) + x.shape)
    for G in Gs:
        # taking partial results left-to-right
        # y = numpy.einsum('i j alpha beta, alpha i ...  -> beta ... j', G, y)
        y = numpy.einsum("i j a b, a i ...  -> b ... j", G, y)
    y1 = y.reshape(-1)

    # alternative way
    y = x.reshape(-1)
    for G in Gs:
        i, j, alpha, beta = G.shape
        y = rearrange(y, "(i rest alpha) -> rest (alpha i)", alpha=alpha, i=i)
        y = y @ rearrange(G, "i j alpha beta -> (alpha i) (j beta)")
        y = rearrange(y, "rest (beta j) -> (beta rest j)", beta=beta, j=j)
    y2 = y
    assert numpy.allclose(y1, y2)

    # yet another way
    y = x
    for G in Gs:
        i, j, alpha, beta = G.shape
        y = rearrange(y, "i ... (j alpha) -> ... j (alpha i)", alpha=alpha, i=i)
        y = y @ rearrange(G, "i j alpha beta -> (alpha i) (j beta)")
    y3 = y.reshape(-1)
    assert numpy.allclose(y1, y3)


def test_pytorch_yolo_fragment():
    if not is_backend_tested("torch"):
        pytest.skip()

    import torch

    def old_way(input, num_classes, num_anchors, anchors, stride_h, stride_w):
        # https://github.com/BobLiu20/YOLOv3_PyTorch/blob/c6b483743598b5f64d520d81e7e5f47ba936d4c9/nets/yolo_loss.py#L28-L44
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in anchors]

        prediction = input.view(bs, num_anchors, 5 + num_classes, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # https://github.com/BobLiu20/YOLOv3_PyTorch/blob/c6b483743598b5f64d520d81e7e5f47ba936d4c9/nets/yolo_loss.py#L70-L92
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # Calculate offsets for each grid
        grid_x = (
            torch.linspace(0, in_w - 1, in_w)
            .repeat(in_w, 1)
            .repeat(bs * num_anchors, 1, 1)
            .view(x.shape)
            .type(FloatTensor)
        )
        grid_y = (
            torch.linspace(0, in_h - 1, in_h)
            .repeat(in_h, 1)
            .t()
            .repeat(bs * num_anchors, 1, 1)
            .view(y.shape)
            .type(FloatTensor)
        )
        # Calculate anchor w, h
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
        # Results
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat(
            (pred_boxes.view(bs, -1, 4) * _scale, conf.view(bs, -1, 1), pred_cls.view(bs, -1, num_classes)), -1
        )
        return output

    def new_way(input, num_classes, num_anchors, anchors, stride_h, stride_w):
        raw_predictions = rearrange(input, " b (anchor prediction) h w -> prediction b anchor h w", anchor=num_anchors)

        anchors = torch.FloatTensor(anchors).to(input.device)
        anchor_sizes = rearrange(anchors, "anchor dim -> dim () anchor () ()")

        _, _, _, in_h, in_w = raw_predictions.shape
        grid_h = rearrange(torch.arange(in_h).float(), "h -> () () h ()").to(input.device)
        grid_w = rearrange(torch.arange(in_w).float(), "w -> () () () w").to(input.device)

        predicted_bboxes = torch.zeros_like(raw_predictions)
        predicted_bboxes[0] = (raw_predictions[0].sigmoid() + grid_h) * stride_h  # center y
        predicted_bboxes[1] = (raw_predictions[1].sigmoid() + grid_w) * stride_w  # center x
        predicted_bboxes[2:4] = (raw_predictions[2:4].exp()) * anchor_sizes  # bbox width and height
        predicted_bboxes[4] = raw_predictions[4].sigmoid()  # confidence
        predicted_bboxes[5:] = raw_predictions[5:].sigmoid()  # class predictions
        # only to match results of original code, not needed
        return rearrange(predicted_bboxes, "prediction b anchor h w -> b anchor h w prediction")

    stride_h = 4
    stride_w = 4
    batch_size = 5
    num_classes = 12
    anchors = [[50, 100], [100, 50], [75, 75]]
    num_anchors = len(anchors)

    input = torch.randn([batch_size, num_anchors * (5 + num_classes), 1, 1])
    result1 = old_way(
        input=input,
        num_anchors=num_anchors,
        num_classes=num_classes,
        stride_h=stride_h,
        stride_w=stride_w,
        anchors=anchors,
    )
    result2 = new_way(
        input=input,
        num_anchors=num_anchors,
        num_classes=num_classes,
        stride_h=stride_h,
        stride_w=stride_w,
        anchors=anchors,
    )
    result1 = result1.reshape(result2.shape)
    assert torch.allclose(result1, result2)
