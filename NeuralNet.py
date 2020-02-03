# -*- coding:utf-8 -*-

import numpy as np
import logging
import copy
from mltools import EpochCalc

class NeuralNet:
    def __init__(self, layers, validate=None):
        self.layers = layers
        self.validate = validate
        if not self.validate:
            self.validate = lambda *args: None
    
    def forward(self, input):
        for l in self.layers:
            input = l.forward(input)
        return input
    
    def train(self, train_data, test_data, epoch, minibatch_size):
        ec = EpochCalc(epoch, len(train_data), minibatch_size)
        
        self.validate(0, self, train_data, test_data)
        for i in range(1, ec.train_update+1):
            mini_data, mini_target = train_data.minibatch(minibatch_size)
            data = copy.deepcopy(mini_data)
            forwards = [mini_data]
            deltas = []

            for l in self.layers:
                data = l.forward(data, cache_signal=True)
                forwards.append(data)

            for n,l in enumerate(self.layers[::-1]):
                if n==0:
                    delta, data = l.back(data, mini_target)
                    deltas.append(delta)
                else:
                    delta, data = l.back(data)
                    deltas.append(delta)
            
            for l, f, d in zip( self.layers, forwards, deltas[::-1] ):
                l.train(f,d)

            if i % ec.epoch_to_update(1) == 0:
                logging.info("[ {} / {} ]( {} / {} )".format(ec.update_to_epoch(i, force_integer=False), ec.train_epoch, i, ec.train_update))
                self.validate(i, self, train_data, test_data)