#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
import matplotlib.pyplot as plt
from mltools.data import Data
from NeuralNet import NeuralNet

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
np.seterr(over="raise", invalid="raise")

plt.rcParams.update({
    "font.family": "IPAGothic",
    "font.size": 20
})

def load_mnist(filename):
    data = np.load(filename)
    data = data.astype("float64") / 255.0
    d = Data(data)
    return d

def main():
    one = load_mnist("./data/one.npy")
    zero = load_mnist("./data/zero.npy")

    nn = NeuralNet.load("./models/one.pickle")
    
    # "0"の再構成
    recon = nn.forward(zero.data[0])
    plt.imshow(zero.data[0].reshape(28, 28), cmap="gray")
    plt.show()
    plt.imshow(recon.reshape(28, 28), cmap="gray")
    plt.show()

    # "1"の再構成
    recon = nn.forward(one.data[0])
    plt.imshow(one.data[0].reshape(28, 28), cmap="gray")
    plt.show()
    plt.imshow(recon.reshape(28, 28), cmap="gray")
    plt.show()

    recon_all = nn.forward(zero.data)
    zero_recon_error = np.mean((zero.data - recon_all)**2, axis=1)
    recon_all = nn.forward(one.data)
    one_recon_error = np.mean((one.data - recon_all)**2, axis=1)
    
    plt.xlabel("データ再構成の二乗誤差")
    plt.ylabel("割合")
    ratio, bins = np.histogram(zero_recon_error, bins=100, range=(0, 0.3))
    ratio = ratio / np.sum(ratio)
    plt.bar(bins[:-1], ratio, width=0.002, alpha=0.7, label="zero")
    ratio, bins = np.histogram(one_recon_error, bins=100, range=(0, 0.3))
    ratio = ratio / np.sum(ratio)
    plt.bar(bins[:-1], ratio, width=0.002, alpha=0.7, label="one")
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()