# -*- coding:utf-8 -*-

import numpy as np

class ReLU:
    @staticmethod
    def activate(input):
        return np.piecewise(input, [input>0], [
            lambda x: x,
            0
        ])
    
    @staticmethod
    def grad(input):
        return np.where(input>0, 1, 0)

class Identity:
    @staticmethod
    def activate(input):
        return input
    
    @staticmethod
    def grad(input):
        return np.ones(input.shape)

def SoftMax(input):
    energy_max = np.max( input, axis=1, keepdims=True)
    energy_exp = np.exp( input - energy_max )
    return energy_exp / np.sum(energy_exp, axis=1, keepdims=True)