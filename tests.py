from einops import transpose
import torch

x = torch.arange(8).reshape(2, 4)
result1 = transpose(x, 'a(bc)->bca', c=2)

x = torch.zeros(10, 20, 30, 40)

y = transpose(x, 'bhwc->bchw')
print(y.shape)


y = transpose(x, 'bhwc->bc(hw)')
print(y.shape)


y = transpose(x, 'bhw(ch1w1)->b(hh1)(ww1)c', h1=2, w1=2)
print(y.shape)


y = transpose(x, 'b(h,h1)(w,w1)c->bhw(h1w1c)', h1=2, w1=2)
print(y.shape)


y1, y2 = transpose(x, 'bhw(cg)->gbhwc', g=2)
print(y1.shape, y2.shape)


y = transpose(x, 'b1sb2t->b1b2st')
print(y.shape)


t = transpose(x, 'bchw->(bhw)c') @ torch.randn(20, 50) # @ это просто перемножение матриц
print(t.shape)

y = transpose(t, '(bhw)c2->bc2hw', b_hw=x.shape)
print(y.shape)


y = transpose(t, '(bhw)c2->bc2hw', b=30, h=10)
print(y.shape)