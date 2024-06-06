import numpy as np
import torch
from torch.autograd import Function

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

class AdditivePerturbation(Function):
    @staticmethod
    def forward(ctx, input):
        n = input.size()[1]
        s = input + np.random.normal(0,1,n)
        ctx.save_for_backward(input)
        return (s>0).clone().detach().to(torch.double).requires_grad_(True)
        return torch.tensor((s > 0).clone().detach(), requires_grad=True, dtype=torch.double)
    @staticmethod
    def backward(ctx, grad_output):
        #print("in AdditivePerturbation grad_output : ")
        #print(grad_output)
        input, = ctx.saved_tensors
        n = input.size()[1]
        samples = []
        for i in range(50):
            noise = np.random.normal(0,1,n)
            s = input + noise > 0
            #print(s)
            #print(s.shape)
            #print(noise.shape)
            samples.append(np.dot(s.reshape(n,1), noise.reshape(1,n)))
            #print(np.shape(samples[-1]))
        #print(samples)
        #print(np.mean(samples, axis=0))
        #print(grad_output * np.mean(samples, axis=0))
        return grad_output * np.mean(samples, axis=0)



# Use it by calling the apply method:
x = torch.tensor([[1.0,2.0]], requires_grad=True, dtype=torch.double)
print(x)
output = Exp.apply(2 * x)
output.retain_grad()
print(output)
print(output.grad)
output.backward(torch.tensor([[0.0, 1.1]]))
print(output.grad)


d = torch.nn.Linear(2, 1, dtype=torch.double)
print(d)
print(d.weight, d.bias)
o = d(x)
print("o = ", o)
o.backward()
print("d.weight.grad = ", d.weight.grad)
print("d.bias.grad = ", d.bias.grad)

print(torch.autograd.gradcheck(d, x))

d = torch.nn.Linear(3,2, dtype=torch.double)
x = torch.tensor([[0.5,0.1,0.0]], requires_grad=True, dtype=torch.double)
x = d(x)
oo = AdditivePerturbation.apply(x)
print(oo)
qq = torch.sum(oo)
print(qq)
qq.backward()

print("d.weight.grad = ", d.weight.grad)
print("d.bias.grad = ", d.bias.grad)

