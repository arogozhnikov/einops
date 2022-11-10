
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

<!-- this link magically rendered as video, unfortunately not in docs -->

https://user-images.githubusercontent.com/6318811/177030658-66f0eb5d-e136-44d8-99c9-86ae298ead5b.mp4




# einops 
[![Run tests](https://github.com/arogozhnikov/einops/actions/workflows/run_tests.yml/badge.svg)](https://github.com/arogozhnikov/einops/actions/workflows/run_tests.yml)
[![PyPI version](https://badge.fury.io/py/einops.svg)](https://badge.fury.io/py/einops)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://einops.rocks/)
![Supported python versions](https://raw.githubusercontent.com/arogozhnikov/einops/master/docs/resources/python_badge.svg)


Flexible and powerful tensor operations for readable and reliable code. <br />
Supports numpy, pytorch, tensorflow, jax, and [others](#supported-frameworks).

## Recent updates:

- einops 0.6 introduces [packing and unpacking](https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb)
- einops 0.5: einsum is now a part of einops
- [Einops paper](https://openreview.net/pdf?id=oapKSVM2bcj) is accepted for oral presentation at ICLR 2022 (yes, it worth reading)
- flax and oneflow backend added
- torch.jit.script is supported for pytorch layers
- powerful EinMix added to einops. [Einmix tutorial notebook](https://github.com/arogozhnikov/einops/blob/master/docs/3-einmix-layer.ipynb) 

<!--<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_logo_350x350.png" 
  alt="einops package logo" width="250" height="250" />
  <br><br>
</div> -->

## Tweets 

> In case you need convincing arguments for setting aside time to learn about einsum and einops...
[Tim Rocktäschel, FAIR](https://twitter.com/_rockt/status/1230818967205425152)

> Writing better code with PyTorch and einops 👌
[Andrej Karpathy, AI at Tesla](https://twitter.com/karpathy/status/1290826075916779520)

> Slowly but surely, einops is seeping in to every nook and cranny of my code. If you find yourself shuffling around bazillion dimensional tensors, this might change your life
[Nasim Rahaman, MILA (Montreal)](https://twitter.com/nasim_rahaman/status/1216022614755463169)

[More testimonials](https://einops.rocks/pages/testimonials/)

## Recordings of talk at ICLR 2022

<a href='https://iclr.cc/virtual/2022/oral/6603'>
<img width="922" alt="Screen Shot 2022-07-03 at 1 00 15 AM" src="https://user-images.githubusercontent.com/6318811/177030789-89d349bf-ef75-4af5-a71f-609896d1c8d9.png">
</a>

Watch [a 15-minute talk](https://iclr.cc/virtual/2022/oral/6603) focused on main problems of standard tensor manipulation methods, and how einops improves this process.


## Contents

- [Installation](#Installation)
- [Documentation](https://einops.rocks/)
- [Tutorial](#Tutorials) 
- [API micro-reference](#API)
- [Why using einops](#Why-using-einops-notation)
- [Supported frameworks](#Supported-frameworks)
- [Contributing](#Contributing)
- [Repository](https://github.com/arogozhnikov/einops) and [discussions](https://github.com/arogozhnikov/einops/discussions)

## Installation  <a name="Installation"></a>

Plain and simple:
```bash
pip install einops
```

<!--
`einops` has no mandatory dependencies (code examples also require jupyter, pillow + backends). 
To obtain the latest github version 

```bash
pip install https://github.com/arogozhnikov/einops/archive/master.zip
```
-->

## Tutorials <a name="Tutorials"></a>

Tutorials are the most convenient way to see `einops` in action

- part 1: [einops fundamentals](https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb) 
- part 2: [einops for deep learning](https://github.com/arogozhnikov/einops/blob/master/docs/2-einops-for-deep-learning.ipynb)
- part 3: [packing and unpacking](https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb)
- part 4: [improve pytorch code with einops](http://einops.rocks/pytorch-examples.html)   


## API <a name="API"></a>

`einops` has a minimalistic yet powerful API.

Three core operations provided ([einops tutorial](https://github.com/arogozhnikov/einops/blob/master/docs/) 
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
And two corresponding layers (`einops` keeps a separate version for each framework) with the same API.

```python
from einops.layers.torch      import Rearrange, Reduce
from einops.layers.tensorflow import Rearrange, Reduce
from einops.layers.flax       import Rearrange, Reduce
from einops.layers.gluon      import Rearrange, Reduce
from einops.layers.keras      import Rearrange, Reduce
from einops.layers.chainer    import Rearrange, Reduce
```

Layers behave similarly to operations and have the same parameters 
(with the exception of the first argument, which is passed during call).

Example of using layers within a model:
```python
# example given for pytorch, but code in other frameworks is almost identical  
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

Later additions to the family are `einsum`, `pack` and `unpack` functions:

```python
from einops import einsum, pack, unpack
# einsum is like ... einsum, generic and flexible dot-product 
# but 1) axes can be multi-lettered  2) pattern goes last 3) works with multiple frameworks
C = einsum(A, B, 'b t1 head c, b t2 head c -> b head t1 t2')

# pack and unpack allow reversibly 'packing' multiple tensors into one.
# Packed tensors may be of different dimensionality:
packed,  ps = pack([class_token_bc, image_tokens_bhwc, text_tokens_btc], 'b * c')
class_emb_bc, image_emb_bhwc, text_emb_btc = unpack(transformer(packed), ps, 'b * c')
# Pack/Unpack are more convenient than concat and split, see tutorial
```

Last, but not the least `EinMix` layer is available! <br />
`EinMix` is a generic linear layer, perfect for MLP Mixers and similar architectures.

## Naming <a name="Naming"></a>

`einops` stands for Einstein-Inspired Notation for operations 
(though "Einstein operations" is more attractive and easier to remember).

Notation was loosely inspired by Einstein summation (in particular by `numpy.einsum` operation).

## Why use `einops` notation?! <a name="Why-using-einops-notation"></a>


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
That's opposed to just writing comments about shapes since comments don't prevent mistakes, not tested, and without code review tend to be outdated   
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

- numpy, cupy, chainer, pytorch: `(60,)`
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

Testimonials provide user's perspective on the same question. 

## Supported frameworks <a name="Supported-frameworks"></a>

Einops works with ...

- [numpy](http://www.numpy.org/)
- [pytorch](https://pytorch.org/)
- [tensorflow](https://www.tensorflow.org/)
- [jax](https://github.com/google/jax)
- [cupy](https://cupy.chainer.org/)
- [chainer](https://chainer.org/)
- [gluon](https://gluon.mxnet.io/)
- [tf.keras](https://www.tensorflow.org/guide/keras)
- [oneflow](https://github.com/Oneflow-Inc/oneflow) (experimental)
- [flax](https://github.com/google/flax) (experimental)

## Citing einops <a name="Contributing"></a>

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

`einops` works with python 3.7 or later.
