# -*- coding:utf-8 -*-

import logging
import numpy as np

def test_error(update_time, model, test_data):
    output = model.forward(test_data.data.data)
    predict = np.argmax(output, axis=1)
    correct = np.sum(predict==test_data.original_answer)
    data_length = len(test_data)
    logging.info("test correct rate: {} ( {} / {} )".format( correct / data_length, correct, data_length ))

def train_error(update_time, model, data):
    output = model.forward(data.data.data)
    predict = np.argmax(output, axis=1)
    correct = np.sum(predict==data.original_answer)
    data_length = len(data)
    logging.info("train correct rate: {} ( {} / {} )".format( correct / data_length, correct, data_length ))