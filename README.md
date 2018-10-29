<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_logo_350x350.png" alt="einops package logo" />
  <br><br>
</div>

# einops

A new flavour of deep learning ops for numpy, pytorch, tensorflow, chainer, gluon, and [others](#supported-frameworks).

`einops` introduces a new way to manipulate tensors, 
providing safer, more readable and semantically richer code.

<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_video.gif" alt="einops package logo" />
  <br><br>
</div>



## Examples

```python
from einops import rearrange, reduce




```


## Layers

Usually it is more convenient to use layers to build models, not operations 
(some frameworks require always using layers).

`Einops` layers are behaving in the same way as operations, and have same parameters

```python
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Rearrange

model = Sequential(
    Conv2d(3, 6, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Conv2d(6, 16, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Rearrange('b c h w -> b (c h w)'),
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 10), 
)
```

Layers are available for `keras`, `torch`, `mxnet` and `gluon`. 

## Naming

`einops` stays for Einstein-Inspired Notion for operations 
(though "Einstein operations" sounds simpler and more attractive).

Notion was loosely inspired by Einstein summation (in particular by `einsum` operation).


## Why using `einops` notion


### Semantic information:

```python
y = x.view(x.shape[0], -1)
y = rearrange(x, 'b c h w -> b (c h w)')
```
while these two lines are doing the same job in some context,
second one provides information about input and output.
Readability also counts.

The next operation looks similar:
```python
y = rearrange(x, 'time c h w -> time (c h w)')
```
it gives reader important information: 
this is not an independent batch of images we are processing, 
but rather a sequence (video). 

Semantic information makes code easier to read and maintain. 

### More checks

Back to the same example:
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)')
```
second line checks that there are four dimensions in input, 
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
# depth to space
rearrange(x, 'b c (h h2) (w w2) -> b (c h2 w2) h w', h2=2, w2=2)
rearrange(x, 'b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=2, w2=2)
```
there are at least four more ways to do it. Which one is used by the framework?

Usually this makes no difference, but it can make a big difference 
(e.g. if you use grouped convolutions on the next stage), and you'd 
like to specify this in your code.

<!-- TODO same with 1d elements -->

### Uniformity

```python
reduce(x, 'b c (x dx) -> b c x', 'max', dx=2)
reduce(x, 'b c (x dx) (y dx) -> b c x y', 'max', dx=2, dy=3)
reduce(x, 'b c (x dx) (y dx) (z dz)-> b c x y z', 'max', dx=2, dy=3, dz=4)
```
This examples demonstrated that there is no need for separate operations for 1d/2d/3d pooling, 
those all are defined in a uniform way. 


Space-to-depth and depth-to space are defined in many frameworks. How about width-to-height?
```python
rearrange(x, 'b c h (w w2) -> b c (h w2) w', w2=2)
```

### Framework independent behavior

Even simple functions may be determined differently

```python
y = x.flatten() # or flatten(x)
```

Suppose `x` shape was `(3, 4, 5)`, then `y` has shape ...
- numpy, cupy, chainer: `(60,)`
- keras, tensorflow.layers, mxnet and gluon: `(3, 20)`
- pytorch: no such function


## Installation

Plain and simple:

```bash
$ pip install einops
```

`einops` has no mandatory dependencies.
 
To obtain the latest github version 
```bash
pip install https://github.com/arogozhnikov/einops/archive/master.zip
```


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
- *prepare a guide/post* for your favorite deep learning framework
- translating examples in languages other than English is also a good idea 
- if you have an educative example, not yet covered by documentation and examples, let me know
- use `einops` notion in your papers to strictly define an operation you're using

## Supported python versions

`einops` works with python 3.5 or later. 

There is nothing specific to python 3 in the code, 
we simply need to move further and I decided not to support python 2.
