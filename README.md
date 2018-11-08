<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_logo_350x350.png" 
  alt="einops package logo" />
  <br><br>
</div>

# einops

Flexible and powerful tensor operations for readable and reliable code. 
Supports numpy, pytorch, tensorflow, and [others](#supported-frameworks).
Examples: 

<a href='http://arogozhnikov.github.io/images/einops/einops_video.mp4' >
<div align="center">
  <br><br>
  <img src="http://arogozhnikov.github.io/images/einops/einops_video.gif" alt="einops package examples" />
  <br>
  <small><a href='http://arogozhnikov.github.io/images/einops/einops_video.mp4'>This video in better quality.</a></small>
  <br><br>
</div>
</a>

- [Tutorial](#Tutorial--Documentation) 
- [API micro-reference](#API)
- [Installation](#Installation)
- [Naming](#Naming-and-terminology)
- [Why using einops](#Why-using-einops-notation)
- [Contributing](#Contributing)
- [Github repository (for issues/questions)](https://github.com/arogozhnikov/einops)


## Tutorial / Documentation 

Tutorial is the most convenient way to see `einops` in action (and right now works as a documentation)

- part 1: [einops fundamentals](https://github.com/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb) 
- part 2: [einops for deep learning](https://github.com/arogozhnikov/einops/blob/master/docs/2-einops-for-deep-learning.ipynb)
- part 3: TBD  

## Installation

Plain and simple:
```bash
pip install einops
```

`einops` has no mandatory dependencies (code examples also require jupyter, pillow + backends). 
To obtain the latest github version 

```bash
pip install https://github.com/arogozhnikov/einops/archive/master.zip
```

## API 

`einops` has minimalistic and powerful API.

Two operations provided (see [einops tutorial](https://github.com/arogozhnikov/einops/blob/master/docs/) for examples)
```python
from einops import rearrange, reduce
# rearrange elements according to the pattern
output_tensor = rearrange(input_tensor, 't b c -> b c t')
# combine rearrangement and reduction
output_tensor = reduce(input_tensor, 'b c (h h2) (w w2) -> b h w c', 'mean', h2=2, w2=2)
```
And two corresponding layers (`einops` keeps separate version for each framework) with the same API.

```python
from einops.layers.chainer import Rearrange, Reduce
from einops.layers.gluon import Rearrange, Reduce
from einops.layers.keras import Rearrange, Reduce
from einops.layers.torch import Rearrange, Reduce
```

Layers are behaving in the same way as operations and have same parameters 
(for the exception of first argument, which should be passed during call)

```python
layer = Rearrange(pattern, **axes_lengths)
layer = Reduce(pattern, reduction, **axes_lengths)

# apply created layer to a tensor / variable
x = layer(x)
```

Example of using layers within a model:
```python
# example given for pytorch, but code in other frameworks is almost identical  
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Reduce

model = Sequential(
    Conv2d(3, 6, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Conv2d(6, 16, kernel_size=5),
    # combined pooling and flattening
    Reduce('b c (h h2) (w w2) -> b (c h w)', 'max', h2=2, w2=2), 
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 10), 
)
```

Additionally two auxiliary functions provided
```python
from einops import asnumpy, parse_shape
# einops.asnumpy converts tensors of imperative frameworks to numpy
numpy_tensor = asnumpy(input_tensor)
# einops.parse_shape gives a shape of axes of interest 
parse_shape(input_tensor, 'batch _ h w') # e.g {'batch': 64, 'h': 128, 'w': 160}
```

## Naming and terminology

`einops` stays for Einstein-Inspired Notation for operations 
(though "Einstein operations" is more attractive and easier to remember).

Notation was loosely inspired by Einstein summation (in particular by `numpy.einsum` operation).

- Terms `tensor` and `ndarray` are equivalently used and refer to multidimensional array 
- Terms `axis` and `dimension` are also equivalent


## Why using `einops` notation


### Semantic information (being verbose in expectations)

```python
y = x.view(x.shape[0], -1)
y = rearrange(x, 'b c h w -> b (c h w)')
```
while these two lines are doing the same job in some context,
second one provides information about input and output.
In other words, `einops` focuses on interface: *what is input and output*, not *how* output is computed.

The next operation looks similar:
```python
y = rearrange(x, 'time c h w -> time (c h w)')
```
But it gives reader a hint: 
this is not an independent batch of images we are processing, 
but rather a sequence (video). 

Semantic information makes code easier to read and maintain. 

### More checks

Reconsider the same example:
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)')
```
second line checks that input has four dimensions, 
but you can also specify particular dimensions. 
That's opposed to just writing comments about shapes since 
[comments don't work](https://medium.freecodecamp.org/code-comments-the-good-the-bad-and-the-ugly-be9cc65fbf83)
as we know   
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)', c=256, h=19, w=19)
```

### Result is strictly determined

Below we have at least two ways to define depth-to-space operation
```python
# depth-to-space
rearrange(x, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2=2, w2=2)
rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=2, w2=2)
```
there are at least four more ways to do it. Which one is used by the framework?

These details are ignored, since *usually* it makes no difference, 
but it can make a big difference (e.g. if you use grouped convolutions on the next stage), 
and you'd like to specify this in your code.

<!-- TODO add example with 1d elements? -->

### Uniformity

```python
reduce(x, 'b c (x dx) -> b c x', 'max', dx=2)
reduce(x, 'b c (x dx) (y dx) -> b c x y', 'max', dx=2, dy=3)
reduce(x, 'b c (x dx) (y dx) (z dz)-> b c x y z', 'max', dx=2, dy=3, dz=4)
```
These examples demonstrated that we don't use separate operations for 1d/2d/3d pooling, 
those all are defined in a uniform way. 

Space-to-depth and depth-to space are defined in many frameworks. But how about width-to-height?
```python
rearrange(x, 'b c h (w w2) -> b c (h w2) w', w2=2)
```

### Framework independent behavior

Even simple functions are defined differently by different frameworks

```python
y = x.flatten() # or flatten(x)
```

Suppose `x` shape was `(3, 4, 5)`, then `y` has shape ...
- numpy, cupy, chainer: `(60,)`
- keras, tensorflow.layers, mxnet and gluon: `(3, 20)`
- pytorch: no such function


## Supported frameworks

Einops works with ...

- [numpy](http://www.numpy.org/)
- [pytorch](https://pytorch.org/)
- [tensorflow eager](https://www.tensorflow.org/guide/eager)
- [cupy](https://cupy.chainer.org/)
- [chainer](https://chainer.org/)
- [gluon](https://mxnet.apache.org/)
- [tensorflow](https://www.tensorflow.org/)
- [mxnet](https://gluon.mxnet.io/) (experimental)
- and [keras](https://keras.io/) (experimental)

## Contributing 

Best ways to contribute are

- spread the word about `einops`
- **prepare a guide/post/tutorial** for your favorite deep learning framework
- translating examples in languages other than English is also a good idea 
- use `einops` notation in your papers to strictly define used operations

## Supported python versions

`einops` works with python 3.5 or later. 

There is nothing specific to python 3 in the code, 
we simply [need to move further](http://github.com/arogozhnikov/python3_with_pleasure) 
and I decided not to support python 2.
