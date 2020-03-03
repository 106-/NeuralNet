# -*- coding:utf-8 -*-

import numpy as np
import logging
import copy
import dill
from mltools import EpochCalc
from mltools.data import CategoricalData

class NeuralNet:
    def __init__(self, layers, validate=None):
        self.layers = layers
        self.validate = validate
    
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input
    
    def train(self, train_data, test_data, epoch, minibatch_size, learning_log):
        ec = EpochCalc(epoch, len(train_data), minibatch_size)
        
        self.validate(0, self, train_data, test_data, learning_log)
        for i in range(1, ec.train_update+1):
            if isinstance(train_data, CategoricalData):
                mini_data, mini_target = train_data.minibatch(minibatch_size)
            else:
                mini_data = mini_target = train_data.minibatch(minibatch_size)
            data = copy.deepcopy(mini_data)
            forwards = [mini_data]
            deltas = []

            for l in self.layers:
                data = l.forward(data, cache_signal=True)
                forwards.append(data)

            for n,l in enumerate(self.layers[::-1]):
                if n==0:
                    data = l.back(data, mini_target)
                    deltas.append(data)
                else:
                    delta, data = l.back(data)
                    deltas.append(delta)
            
            for l, f, d in zip( self.layers, forwards, deltas[::-1] ):
                l.train(f,d)

            if i % ec.epoch_to_update(1) == 0:
                logging.info("[ {} / {} ]( {} / {} )".format(ec.update_to_epoch(i, force_integer=False), ec.train_epoch, i, ec.train_update))
                if self.validate:
                    self.validate(ec.update_to_epoch(i), self, train_data, test_data, learning_log)
    
    def save(self, filename):
        dill.dump(self, open(filename, "wb+"))
    
    @staticmethod
    def load(filename):
        return dill.load(open(filename, "rb"))