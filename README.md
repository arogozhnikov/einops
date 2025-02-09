
<!--
<a href='http://arogozhnikov.github.io/images/einops/einops_video.mp4' >
<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_video.gif" alt="einops package examples" />
  <br>
  <small><a href='http://arogozhnikov.github.io/images/einops/einops_video.mp4'>This video in high quality (mp4)</a></small>
  <br><br>
</div>
</a>
-->

<!-- this link magically rendered as video on github readme, unfortunately not in docs -->

https://user-images.githubusercontent.com/6318811/177030658-66f0eb5d-e136-44d8-99c9-86ae298ead5b.mp4




# einops 
[![Run tests](https://github.com/arogozhnikov/einops/actions/workflows/run_tests.yml/badge.svg)](https://github.com/arogozhnikov/einops/actions/workflows/run_tests.yml)
[![PyPI version](https://badge.fury.io/py/einops.svg)](https://badge.fury.io/py/einops)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://einops.rocks/)
![Supported python versions](https://raw.githubusercontent.com/arogozhnikov/einops/main/docs/resources/python_badge.svg)


Flexible and powerful tensor operations for readable and reliable code. <br />
Supports numpy, pytorch, tensorflow, jax, and [others](#supported-frameworks).

## Recent updates:

- 0.8.0: tinygrad backend added, small fixes
- 0.7.0: no-hassle `torch.compile`, support of [array api standard](https://data-apis.org/array-api/latest/API_specification/index.html) and more
- 10'000🎉: github reports that more than 10k project use einops
- einops 0.6.1: paddle backend added
- einops 0.6 introduces [packing and unpacking](https://github.com/arogozhnikov/einops/blob/main/docs/4-pack-and-unpack.ipynb)
- einops 0.5: einsum is now a part of einops
- [Einops paper](https://openreview.net/pdf?id=oapKSVM2bcj) is accepted for oral presentation at ICLR 2022 (yes, it worth reading).
  Talk recordings are [available](https://iclr.cc/virtual/2022/oral/6603)


<details markdown="1">
<summary>Previous updates</summary>
- flax and oneflow backend added
- torch.jit.script is supported for pytorch layers
- powerful EinMix added to einops. [Einmix tutorial notebook](https://github.com/arogozhnikov/einops/blob/main/docs/3-einmix-layer.ipynb) 
</details>

<!--<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_logo_350x350.png" 
  alt="einops package logo" width="250" height="250" />
  <br><br>
</div> -->


## Tweets 

> In case you need convincing arguments for setting aside time to learn about einsum and einops...
[Tim Rocktäschel](https://twitter.com/_rockt/status/1230818967205425152)

> Writing better code with PyTorch and einops 👌
[Andrej Karpathy](https://twitter.com/karpathy/status/1290826075916779520)

> Slowly but surely, einops is seeping in to every nook and cranny of my code. If you find yourself shuffling around bazillion dimensional tensors, this might change your life
[Nasim Rahaman](https://twitter.com/nasim_rahaman/status/1216022614755463169)

[More testimonials](https://einops.rocks/pages/testimonials/)


## Contents

- [Installation](#Installation)
- [Documentation](https://einops.rocks/)
- [Tutorial](#Tutorials)
- [API micro-reference](#API)
- [Why use einops](#Why-use-einops-notation)
- [Supported frameworks](#Supported-frameworks)
- [Citing](#Citing)
- [Repository](https://github.com/arogozhnikov/einops) and [discussions](https://github.com/arogozhnikov/einops/discussions)

## Installation  <a name="Installation"></a>

Plain and simple:
```bash
pip install einops
```

## Tutorials <a name="Tutorials"></a>

Tutorials are the most convenient way to see `einops` in action

- part 1: [einops fundamentals](https://github.com/arogozhnikov/einops/blob/main/docs/1-einops-basics.ipynb)
- part 2: [einops for deep learning](https://github.com/arogozhnikov/einops/blob/main/docs/2-einops-for-deep-learning.ipynb)
- part 3: [packing and unpacking](https://github.com/arogozhnikov/einops/blob/main/docs/4-pack-and-unpack.ipynb)
- part 4: [improve pytorch code with einops](http://einops.rocks/pytorch-examples.html)

Kapil Sachdeva recorded a small [intro to einops](https://www.youtube.com/watch?v=xGy75Pjsqzo).

## API <a name="API"></a>

`einops` has a minimalistic yet powerful API.

Three core operations provided ([einops tutorial](https://github.com/arogozhnikov/einops/blob/main/docs/)
shows those cover stacking, reshape, transposition, squeeze/unsqueeze, repeat, tile, concatenate, view and numerous reductions)

```python
from einops import rearrange, reduce, repeat
# rearrange elements according to the pattern
output_tensor = rearrange(input_tensor, 't b c -> b c t')
# combine rearrangement and reduction
output_tensor = reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
# copy along a new axis
output_tensor = repeat(input_tensor, 'h w -> h w c', c=3)
```

Later additions to the family are `pack` and `unpack` functions (better than stack/split/concatenate):

```python
from einops import pack, unpack
# pack and unpack allow reversibly 'packing' multiple tensors into one.
# Packed tensors may be of different dimensionality:
packed,  ps = pack([class_token_bc, image_tokens_bhwc, text_tokens_btc], 'b * c')
class_emb_bc, image_emb_bhwc, text_emb_btc = unpack(transformer(packed), ps, 'b * c')
```

Finally, einops provides einsum with a support of multi-lettered names:

```python
from einops import einsum, pack, unpack
# einsum is like ... einsum, generic and flexible dot-product
# but 1) axes can be multi-lettered  2) pattern goes last 3) works with multiple frameworks
C = einsum(A, B, 'b t1 head c, b t2 head c -> b head t1 t2')
```

### EinMix

`EinMix` is a generic linear layer, perfect for MLP Mixers and similar architectures.

### Layers

Einops provides layers (`einops` keeps a separate version for each framework) that reflect corresponding functions

```python
from einops.layers.torch      import Rearrange, Reduce
from einops.layers.tensorflow import Rearrange, Reduce
from einops.layers.flax       import Rearrange, Reduce
from einops.layers.paddle     import Rearrange, Reduce
```

<details markdown="1">
<summary>Example of using layers within a pytorch model</summary>
Example given for pytorch, but code in other frameworks is almost identical

```python 
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Rearrange

model = Sequential(
    ...,
    Conv2d(6, 16, kernel_size=5),
    MaxPool2d(kernel_size=2),
    # flattening without need to write forward
    Rearrange('b c h w -> b (c h w)'),
    Linear(16*5*5, 120),
    ReLU(),
    Linear(120, 10),
)
```

No more flatten needed!

Additionally, torch layers as those are script-able and compile-able.
Operations [are torch.compile-able](https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops),
 but not script-able due to limitations of torch.jit.script.
</details>




## Naming <a name="Naming"></a>

`einops` stands for Einstein-Inspired Notation for operations 
(though "Einstein operations" is more attractive and easier to remember).

Notation was loosely inspired by Einstein summation (in particular by `numpy.einsum` operation).

## Why use `einops` notation?! <a name="Why-use-einops-notation"></a>


### Semantic information (being verbose in expectations)

```python
y = x.view(x.shape[0], -1)
y = rearrange(x, 'b c h w -> b (c h w)')
```
While these two lines are doing the same job in *some* context,
the second one provides information about the input and output.
In other words, `einops` focuses on interface: *what is the input and output*, not *how* the output is computed.

The next operation looks similar:

```python
y = rearrange(x, 'time c h w -> time (c h w)')
```
but it gives the reader a hint:
this is not an independent batch of images we are processing,
but rather a sequence (video).

Semantic information makes the code easier to read and maintain.

### Convenient checks

Reconsider the same example:

```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)')
```
The second line checks that the input has four dimensions,
but you can also specify particular dimensions.
That's opposed to just writing comments about shapes since comments don't prevent mistakes,
not tested, and without code review tend to be outdated
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)', c=256, h=19, w=19)
```

### Result is strictly determined

Below we have at least two ways to define the depth-to-space operation
```python
# depth-to-space
rearrange(x, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2=2, w2=2)
rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=2, w2=2)
```
There are at least four more ways to do it. Which one is used by the framework?

These details are ignored, since *usually* it makes no difference,
but it can make a big difference (e.g. if you use grouped convolutions in the next stage),
and you'd like to specify this in your code.


### Uniformity

```python
reduce(x, 'b c (x dx) -> b c x', 'max', dx=2)
reduce(x, 'b c (x dx) (y dy) -> b c x y', 'max', dx=2, dy=3)
reduce(x, 'b c (x dx) (y dy) (z dz) -> b c x y z', 'max', dx=2, dy=3, dz=4)
```
These examples demonstrated that we don't use separate operations for 1d/2d/3d pooling,
those are all defined in a uniform way.

Space-to-depth and depth-to space are defined in many frameworks but how about width-to-height? Here you go:

```python
rearrange(x, 'b c h (w w2) -> b c (h w2) w', w2=2)
```


### Framework independent behavior

Even simple functions are defined differently by different frameworks

```python
y = x.flatten() # or flatten(x)
```

Suppose `x`'s shape was `(3, 4, 5)`, then `y` has shape ...

- numpy, pytorch, cupy, chainer, jax: `(60,)`
- keras, tensorflow.layers, gluon: `(3, 20)`

`einops` works the same way in all frameworks.


### Independence of framework terminology

Example: `tile` vs `repeat` causes lots of confusion. To copy image along width:
```python
np.tile(image, (1, 2))    # in numpy
image.repeat(1, 2)        # pytorch's repeat ~ numpy's tile
```

With einops you don't need to decipher which axis was repeated:
```python
repeat(image, 'h w -> h (tile w)', tile=2)  # in numpy
repeat(image, 'h w -> h (tile w)', tile=2)  # in pytorch
repeat(image, 'h w -> h (tile w)', tile=2)  # in tf
repeat(image, 'h w -> h (tile w)', tile=2)  # in jax
repeat(image, 'h w -> h (tile w)', tile=2)  # in cupy
... (etc.)
```

[Testimonials](https://einops.rocks/pages/testimonials/) provide users' perspective on the same question.


## Supported frameworks <a name="Supported-frameworks"></a>

Einops works with ...

- [numpy](http://www.numpy.org/)
- [pytorch](https://pytorch.org/)
- [tensorflow](https://www.tensorflow.org/)
- [jax](https://github.com/google/jax)
- [cupy](https://github.com/cupy/cupy)
- [flax](https://github.com/google/flax) (community)
- [paddle](https://github.com/PaddlePaddle/Paddle) (community)
- [oneflow](https://github.com/Oneflow-Inc/oneflow) (community)
- [tinygrad](https://github.com/tinygrad/tinygrad) (community)
- [pytensor](https://github.com/pymc-devs/pytensor) (community)

Additionally, einops can be used with any framework that supports
[Python array API standard](https://data-apis.org/array-api/latest/API_specification/index.html),
which includes

- numpy >= 2.0
- [MLX](https://github.com/ml-explore/mlx)
- [pydata/sparse](https://github.com/pydata/sparse) >= 0.15
- [quantco/ndonnx](https://github.com/Quantco/ndonnx)
- recent releases of jax and cupy.
- dask is supported via [array-api-compat](https://github.com/data-apis/array-api-compat)


## Development

Devcontainer is provided, this environment can be used locally, or on your server,
or within github codespaces. 
To start with devcontainers in vs code, clone repo, and click 'Reopen in Devcontainer'. 

Starting from einops 0.8.1, einops distributes tests as a part of package.

```bash
# pip install einops pytest
python -m einops.tests.run_tests numpy pytorch jax --pip-install
```

`numpy pytorch jax` is an _example_, any subset of testable frameworks can be provided.
Every framework is tested against numpy, so it is a requirement for tests.

Specifying `--pip-install` will install requirements in current virtualenv,
and should be omitted if dependencies are installed locally.

To build/test docs:

```bash
hatch run docs:serve  # Serving on http://localhost:8000/
```


## Citing einops <a name="Citing"></a>

Please use the following bibtex record

```text
@inproceedings{
    rogozhnikov2022einops,
    title={Einops: Clear and Reliable Tensor Manipulations with Einstein-like Notation},
    author={Alex Rogozhnikov},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=oapKSVM2bcj}
}
```


## Supported python versions

`einops` works with python 3.8 or later.
