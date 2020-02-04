# -*- coding:utf-8 -*-

import numpy as np
from ActivationFunctions import SoftMax
from mltools import EpochCalc
from mltools import Parameter

class Dense:
    def __init__(self, size, activation_func, optimizer):
        self.activation_func = activation_func
        self.optimizer = optimizer
        self.params = DenseParams(size)
        self.size = size
    
    def forward(self, lower_input, cache_signal=False):
        signal = np.dot( lower_input, self.params.weight ) + self.params.bias
        if cache_signal:
            self._last_signal = signal 
        return self.activation_func.activate(signal)

    def back(self, upper_input):
        delta = self.activation_func.grad(self._last_signal) * upper_input
        next_input = np.dot(delta, self.params.weight.T)
        return delta, next_input
    
    def train(self, output, delta):
        grad_weight = np.dot( output.T, delta ) / output.shape[0]
        grad_bias = np.mean(delta, axis=0)
        diff = self.optimizer.update(DenseParams(self.size, [grad_bias, grad_weight]))
        self.params += diff

class SoftMaxLayer:
    def __init__(self):
        pass

    def forward(self, lower_input, **kwargs):
        return SoftMax(lower_input)

    def back(self, upper_input, target):
        return target - upper_input

    def train(self, *args):
        pass

class MeanSquareErrorLayer:
    def __init__(self):
        pass

    def forward(self, lower_input, **kwargs):
        return lower_input
    
    def back(self, upper_input, target):
        return target - upper_input
    
    def train(self, *args):
        pass

class DenseParams(Parameter):
    names = ["bias", "weight"]

    def __init__(self, size, initial_params=None):
        self.size = size

        if not initial_params:
            uniform_range = np.sqrt( 6/sum(size) )
            self.weight = np.random.uniform( -uniform_range, uniform_range, size )
            self.bias = np.random.randn(size[1])
        
        elif isinstance(initial_params, dict):
            for i in self.names:
                setattr(self, i, initial_params[i])
        
        elif isinstance(initial_params, list):
            for i,j in zip(self.names, initial_params):
                setattr(self, i, j)

        params = {}
        for i in self.names:
            params[i] = getattr(self, i)
        super().__init__(params)
    
    def zeros(self):
        zero_params = {}
        for i in self.names:
            zero_params[i] = np.zeros(self[i].shape)
        return DenseParams(self.size, initial_params=zero_params)