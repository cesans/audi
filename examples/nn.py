# coding: utf-8

__author__ = "Carlos Sanchez @cesans"


from pyaudi import gdual
from pyaudi import sin, cos, tanh

import numpy as np


def initialize_weights(n_units, order, m=1):
    """
    Returns a list with the gduals for the weights of NN initialized with values from a normal(0,m) distribution

    :param n_units: list with the number of units in each layer.
        Eg. [3,2,2,1]: 3 inputs, 2 hidden layers with 2 units and 1 output
    :param order: audi truncation order for the gduals
    :param m: std of the random weights
    :return: list of layers, for each layer list of units, for each unit list of weights
    """
    weights = []

    for layer in range(1, len(n_units)):
        weights.append([])
        for unit in range(n_units[layer]):
            weights[-1].append([])
            for prev_unit in range(n_units[layer-1]):
                symname = 'w_{{({0},{1},{2})}}'.format(layer, unit, prev_unit)
                w = gdual(m*np.random.randn(), symname, order)
                weights[-1][-1].append(w)
    return weights


def initialize_biases(n_units, order, v=1):
    """
    Returns a list with the biases for the weights of NN initialized as v

    :param n_units: list with the number of units in each layer.
        Eg. [3,2,2,1]: 3 inputs, 2 hidden layers with 2 units and 1 output
    :param order: audi truncation order for the gduals
    :param v: value of the biases
    :return: list of layers, for each layer list of biases
    """
    biases = []

    for layer in range(1, len(n_units)):
        biases.append([])
        for unit in range(n_units[layer]):
            symname = 'b_{{({0},{1})}}'.format(layer, unit)
            b = gdual(v, symname, order)
            biases[-1].append(b)
    return biases


def N_f(inputs, w, b):
    """
    Computes the expression for the outputs of a NN
    Returns an
    :param inputs as a list (or numpy)
    :param w: weights
    :param b: biases
    :return: list with the expression of the outputs of the NN
    """

    prev_layer_outputs = inputs

    # Hidden layers
    for layer in range(len(w)):

        this_layer_outputs = []

        for unit in range(len(w[layer])):

            unit_output = 0
            unit_output += b[layer][unit]

            for prev_unit, prev_output in enumerate(prev_layer_outputs):
                unit_output += w[layer][unit][prev_unit] * prev_output

            if layer != len(w)-1:
                unit_output = tanh(unit_output)

            this_layer_outputs.append(unit_output)

        prev_layer_outputs = this_layer_outputs

    return prev_layer_outputs


def GD_update(loss, w, b, lr):
    """
    Updates the weights and biases of a network with gradient descent

    :param loss gdual expression of the loss function
    :param w weights
    :param b biases
    :param lr learning rate
    :return: new (weights, biases)
    """

    # Update weights
    for layer in range(len(w)):
        for unit in range(len(w[layer])):
            for prev_unit in range(len(w[layer][unit])):

                weight = w[layer][unit][prev_unit]
                if weight.symbol_set[0] in loss.symbol_set:
                    symbol_idx = loss.symbol_set.index(weight.symbol_set[0])
                    d_idx = [0]*loss.symbol_set_size
                    d_idx[symbol_idx] = 1

                    # eg. if d_idx = [1,0,0,0,...] get get the derivatives of loss wrt
                    # the first symbol (variable) in loss.symbol_set
                    w[layer][unit][prev_unit] -= loss.get_derivative(d_idx) * lr

    # Update biases
    for i in range(len(b)):
        for j in range(len(b[layer])):

                bias = b[layer][unit]
                if bias.symbol_set[0] in loss.symbol_set:
                    symbol_idx = loss.symbol_set.index(bias.symbol_set[0])
                    d_idx = [0]*loss.symbol_set_size
                    d_idx[symbol_idx] = 1
                    b[layer][unit] -= loss.get_derivative(d_idx) * lr

    return w, b
