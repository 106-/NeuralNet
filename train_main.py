#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
from mltools.optimizer import Adamax
from mltools.data import CategoricalData
from mltools import LearningLog
from NeuralNet import NeuralNet
from ActivationFunctions import ReLU, SoftMax, Identity
from Layers import Dense, SoftMaxLayer
from Validations import test_error, train_error, cross_entrpy

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

def load_mnist(filename):
    target, data = np.split(np.load(filename), [1], axis=1)
    data = data.astype("float64") / 255.0
    ok_target = np.eye(10)[target.astype("int64").flatten().tolist()]
    cd = CategoricalData(data, ok_target)
    cd.original_answer = target.astype("int64").flatten()
    return cd

def main():
    test = load_mnist("./data/mnist_test.npy")
    train = load_mnist("./data/mnist_train.npy")
    learning_log = LearningLog({})

    def validate(epoch, model, train, test, learning_log):
        test_correct = test_error(model, test)
        logging.info("test correct rate: {}".format(test_correct))
        train_correct = train_error(model, train)
        logging.info("train correct rate: {}".format(train_correct))
        ce = cross_entrpy(model, train)
        logging.info("cross entrpy: {}".format(ce))
        learning_log.make_log(epoch, "test-correct", [test_correct])
        learning_log.make_log(epoch, "train-correct", [train_correct])
        learning_log.make_log(epoch, "cross_entropy", [ce])

    nn = NeuralNet([
        Dense((784, 200), ReLU, Adamax()),
        Dense((200, 10), Identity, Adamax()),
        SoftMaxLayer(),
    ], validate)
    nn.train(train, test, 50, 100, learning_log)
    learning_log.save("learning_log.json")

if __name__=='__main__':
    main()