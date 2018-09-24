from einops import transpose
import torch

x = torch.arange(8).reshape(2, 4)
result1 = transpose(x, 'a(bc)->bca', c=2)
assert result1.shape == (2, 2, 2)

x = torch.zeros(10, 20, 30, 40)

y = transpose(x, 'bhwc->bchw')
print(y.shape)
assert y.shape == (10, 40, 20, 30)


y = transpose(x, 'bhwc->bc(hw)')
print(y.shape)
assert y.shape == (10, 40, 20 * 30)


y = transpose(x, 'bhw(ch1w1)->b(hh1)(ww1)c', h1=2, w1=2)
print(y.shape)
assert y.shape == (10, 40, 60, 10)

y = transpose(x, 'b(h,h1)(w,w1)c->bhw(h1w1c)', h1=2, w1=2)
print(y.shape)
assert y.shape == (10, 10, 15, 160)


y1, y2 = transpose(x, 'bhw(cg)->gbhwc', g=2)
print(y1.shape, y2.shape)
assert y1.shape == (10, 20, 30, 20)
assert y2.shape == (10, 20, 30, 20)


y = transpose(x, 'b1sb2t->b1b2st')
print(y.shape)
assert y.shape == (10, 30, 20, 40)

t = transpose(x, 'bchw->(bhw)c') @ torch.randn(20, 50) # @ это просто перемножение матриц
print(t.shape)
assert y.shape == (10 * 30 * 40, 50)

y = transpose(t, '(b h w) c2 -> b c2 h w', b_hw=x.shape)
print(y.shape)
assert y.shape == (10, 50, 30, 40)

y = transpose(t, '(bhw)c2->bc2hw', b=30, h=10)
print(y.shape)
assert y.shape == (30, 50, 10, 40)
