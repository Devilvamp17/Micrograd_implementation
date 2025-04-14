import torch
import random
import numpy as np
import math
class value:
    def __init__(self,data,_children=[],_op=None,label=""): 
        self._op=_op
        self.label=label
        self.grad=0.0
        self._backward=lambda:None
        self._children=set(_children)
        self.data=data
    def __repr__(self):
        return f"value(data={self.data})"
    def __add__(self,other):
        other=other if isinstance(other,value) else value(other)
        out = value(self.data+other.data,(self,other),'+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward=_backward
        return out 
    def __neg__(self):
        return self * -1
    def __sub__(self,other):
        return self + (-other)

    def __mul__(self,other):
        other=other if isinstance(other,value) else value(other)
        out= value(self.data*other.data,( self,other),'*')
        def _backward():
            self.grad += other.grad * out.grad
            other.grad += self.grad * out.grad
        
        out._backward=_backward
        return out 
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out=value(self.data**other,(self,),'^')
        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward=_backward
        return out
    
    def __truediv__(self,other):
        return self*other**-1

    def __rmul__(self,other):
        return self*other

    def tanh(self):
        out=value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward=_backward
        return out
    
    def exp(self):
        out=value(math.exp(self.data),(self,),'exp') 
        def _backward():
            self.grad +=out.data * out.grad
        out._backward=_backward
        return out

    def relu(self):
        out=value(max(0,self.data),(self,), 'relu')
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward=_backward
        return out
    
    def sigmoid(self):
        out=value(1/(1+math.exp(-self.data)),(self,), 'sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward=_backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad=0
    def parameters(self):
        return []

class Neuron:
    def __init__(self,weights,bias):
        self.weights=weights
        self.bias=bias
        self.grad_w=0
        self.grad_b=0
        self._out=None

    def __call__(self,x):
        act = x @ self.weights + self.bias
        out= act.tanh()
        def _backward():
            self.grad_w += (1 - out.data**2) * x * out.grad
            self.grad_b += (1 - out.data**2) * out.grad
        return out
    def parameters(self):
        return self.weights+[self.bias]     
    
class Layer:
    def __init__(self,n_inputs,n_neurons):
        self.neurons=[Neuron(np.random.randn(n_inputs),np.random.randn(1)) for _ in range(n_neurons)]
        self._out=None
    def __call__(self,x):
        outs=[n(x) for n in self.neurons]
        out = value(np.stack(outs),self.neurons)
        def _backward():
            for n in self.neurons:
                n.grad_w=0
                n.grad_b=0
            out.grad=1
            out._backward()
        out._backward=_backward
        return out  
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    def __init__(self,n_inputs,n_outs):
        sz=[n_inputs]+n_outs
        self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(n_outs))]
        self._out=None
    def __call__(self,x):
        for l in self.layers:
            x=l(x)
        return x
    def backward(self):
        for l in self.layers:
            l._out.grad=1
            l._out._backward()
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]