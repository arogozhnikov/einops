logo here

# einops

A new flavour of deep learning ops for pytorch, chainer, gluon, tensorflow and others.

## About

`einops` introduces a new way to manipulate tensors, which is better seen in examples.

## Examples

```python
from einops import rearrange, reduce




```


## Layers

Usually it is more convenient to use layers to build models, not operations 
(some frameworks require using layers)

Layers are behaving in the same way as operations 

```python
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, ReLU
from einops.layers.torch import Rearrange, Reduce
model = Sequential(
    Conv2d(3, 6, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Conv2d(6, 16, kernel_size=5),
    MaxPool2d(kernel_size=2),
    Rearrange('b c h w -> b (c h w)'),
    Linear(16*5*5, 120), 
    ReLU(),
    Linear(120, 84), 
    ReLU(),    
    Linear(84, 10), 
)
```

## Why `einops` notion is good:


### Semantics:

```python
y = x.view(x.shape[0], -1)
y = rearrange(x, 'b c h w -> b (c h w)')
```
while these two lines are doing the same job in the same context,
second one provides information about input and output.
Readability also counts.

t c h w
b c h w 

reduction over time

### More checks

Back to the same example:
```python
y = x.view(x.shape[0], -1) # x: (batch, 256, 19, 19)
y = rearrange(x, 'b c h w -> b (c h w)')
```
at least checks that there are four dimensions in input, 
but you can also specify particular dimensions. 
That's opposed to just writing comments about shapes since 
[comments dont work](https://medium.freecodecamp.org/code-comments-the-good-the-bad-and-the-ugly-be9cc65fbf83)   
```python
y = rearrange(x, 'b c h w -> b (c h w)', c=256, h=19, w=19)
```

### Result is strictly determined:

Below we have at least two ways to define depth-to-space operation
```python
# depth to space
rearrange('b c (h h2) (w w2) -> b (c h2 w2) h w', h2=2, w2=2)
rearrange('b c (h h2) (w w2) -> b (h2 w2 c) h w', h2=2, w2=2)
```
there are at least four more. Which one is used by the framework?

This may have no difference, and it can make a difference 
(e.g. if you use grouped convolutions on the next stage)

<!-- TODO same with 1d elements -->

### Uniformity:
2d max-pooling is defined in the same way as 1d and 3d
space-to-depth ot width-to-height have the same motion

## Installation

Plain and simple:

```bash
$ pip install einops
```

The only dependency `einops` has is `numpy`. 
To obtain the latest version use 
```bash
pip install https://github.com/arogozhnikov/einops/archive/master.zip
```


## Working with ...

- [numpy](http://www.numpy.org/)
- [pytorch](https://pytorch.org/)
- [tensorflow eager](https://www.tensorflow.org/guide/eager)
- [cupy](https://cupy.chainer.org/)
- [chainer](https://chainer.org/)
- [gluon](https://mxnet.apache.org/)
- [mxnet](https://gluon.mxnet.io/)
- [tensorflow](https://www.tensorflow.org/)
- and [keras](https://keras.io/) (experimental)

## Contributing 

Best ways to contribute are

- spread the word about `einops`
- prepare a guide/post specifically for your favorite deep learning framework
- if you have an interesting use case, not yet covered by documentation, let me know


## Supported python versions

`einops` works with python 3.5 or later. 

There is nothing specific to python 3 in the code, just we need to move further.