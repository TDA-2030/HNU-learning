import torch


def f(x):
    '''计算y'''
    y = x**2 * torch.exp(x)
    return y

def gradf(x):
    '''手动求导函数'''
    dx = 2*x*torch.exp(x) + x**2*torch.exp(x)
    return dx

x = torch.randn(3,4, requires_grad = True)
y = f(x)

y.backward(torch.ones(y.size())) # gradient形状与y一致
x.grad