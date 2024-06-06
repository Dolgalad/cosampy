import torch
from torch.autograd import *

class P2(Function):
    @staticmethod
    def forward(ctx, i):
        res = i**2
        ctx.save_for_backward(res)
        return res
    @staticmethod
    def backward(ctx, grad_output):
        res, = ctx.saved_tensors
        print(res)
        return grad_output * 2

class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

x = Variable(torch.Tensor([1,2]), requires_grad=True)

o = Exp.apply(x)
print(o)

o.backward(torch.Tensor([0,1]))
print(o.grad)
