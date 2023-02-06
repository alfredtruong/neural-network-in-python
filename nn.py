# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:03:57 2023

@author: ahkar

https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""

from random import random 
from typing import Dict,List
from math import exp

NODE  = Dict[str,float]
LAYER = List[NODE]
NNET  = List[LAYER]

ATTRS       = List[float]
CLASS       = int
NNET_INPUT  = "ATTRS,CLASS"

CLASS_1HOT  = List[int] # 1-hot encoding of CLASS, transform int (class) into List[int] (class vector)
NNET_OUTPUT = CLASS_1HOT # 1-hot encoding prediction from nnet into class vector

# initialize node
def initialize_node(n_node_inputs : int) -> NODE :
    return {
        'bias'    : random(),
        'weights' : [random() for i in range(n_node_inputs)]
    }

# initialize layer
def initialize_layer(n_layer_inputs : int, n_layer_nodes : int) -> LAYER :
    return [initialize_node(n_layer_inputs) for i in range(n_layer_nodes)]

# initialize network
def initialize_nnet(n_nnet_inputs : int, n_nnet_hidden : int, n_nnet_outputs : int) -> NNET :
    network = []
    network.append(initialize_layer(n_nnet_inputs,n_nnet_hidden))
    network.append(initialize_layer(n_nnet_hidden,n_nnet_outputs))
    return network

# inspect a network
def print_nnet(nnet : NNET) -> None :
    for nnet_layer in nnet:
        print(nnet_layer)
     
'''
# test network setup
nn = initialize_nnet(2,3,2)
print_nnet(nn)
'''

# neuron activation
def activate(bias : float, weights : List[float], inputs : List[float]) -> float :
    activation = bias
    for i in range(len(weights)): # for loops runs till end of weights cos CLASS at end of INPUT and not used
        activation += weights[i] * inputs[i]
    return activation

# neuron transfer
def transfer(activation : float) -> float :
    return 1/(1+exp(-activation))

# forward propagate input through network
def forward_propagate(nnet : NNET, nnet_input : NNET_INPUT) -> NNET_OUTPUT:
    layer_inputs = nnet_input
    for nnet_layer in nnet:
        print(f"inputs \t\t: {layer_inputs}")
        layer_outputs = []
        for layer_node in nnet_layer:
            activation = activate(layer_node['bias'],layer_node['weights'],layer_inputs)
            layer_node['output'] = transfer(activation) # enrich node dico with 'output'
            layer_outputs.append(layer_node['output'])
        print(f"outputs \t: {layer_outputs}")
        print()
        layer_inputs = layer_outputs # prep for next iteration
    return layer_inputs

'''
# test forward propagate
forward_propagate(nn,[2,3])
'''

# node derivative
def transfer_derivative(node_output : float) -> float :
    return node_output * (1 - node_output)

# back propagate error through network
def back_propagate_error(nnet : NNET, nnet_expected : CLASS_1HOT) -> None :
    for i in reversed(range(len(nnet))):
        layer = nnet[i]
        # compile errors this layer needs to see
        errors = []
        if i == len(nnet)-1:
            for j in range(len(layer)): # can use enumerate here
                node = layer[j]
                errors.append(node['output'] - nnet_expected[j])
        else:
            for j in range(len(layer)):
                error = 0;
                for node in nnet[i+1]: # look at forward errors and combine them to now
                    error += node['weights'][j] * node['delta']
                errors.append(error)
        # compute deltas of this layer using errors of this layer
        for j in range(len(layer)):
            node = layer[j]
            node['delta'] = errors[j] * transfer_derivative(node['output']) # enrich node dico with 'delta'

'''
# test back_propagate_error
nn2=\
[[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]

back_propagate_error(nn2,[0,1])

print_nnet(nn2)

[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': 0.0005348048046610517}]
[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': -0.0771723774346327}]

[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': 0.0005348048046610517}]
[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': -0.0771723774346327}]

[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 'delta': 0.0005348048046610517}]
[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095], 'delta': 0.14619064683582808}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763], 'delta': -0.0771723774346327}]
'''

# update network weights on all layers
def update_weights(nnet : NNET, nnet_input : NNET_INPUT, learning_rate : float) -> None :
    for i in range(len(nnet)):
        # get inputs for current layer
        if i == 0:
            # is attrs (NNET_INPUT without CLASS) for input layer
            attrs = nnet_input[-1]
        else:
            # is output from previous layer for non-input layers
            attrs = [node['output'] for node in nnet[i-1]]
        # update weights for all nodes in layer
        for node in nnet[i]:
            # update node bias
            node['bias'] -= learning_rate * node['delta']
            # update node input weighting
            for j in range(attrs):
                node['weights'][j] -= learning_rate * node['delta'] * attrs[j]
    
def class_to_1hot(n_outputs : int, class_idx : CLASS) -> CLASS_1HOT :
    onehot = [0 for i in range(n_outputs)]
    onehot[class_idx] = 1
    return onehot
    
# train a network for a fixed number of epochs
def train_nnet(nnet : NNET, train : None, learning_rate : float, n_epoch : int, n_outputs : int):
    for epoch in range(n_epoch):
        sum_error = 0
        for nnet_input in train:
            # compute nnet error given in put
            nnet_output = forward_propagate(nnet,nnet_input)
            expected = class_to_1hot(n_outputs,nnet_input[-1])
            sum_error += sum((expected[i] - nnet_output[i])**2 for i in range(len(expected))) # increment sum_error
            # back propagate error through nnet
            back_propagate_error(nnet,expected)
            # update nnet weights
            update_weights(nnet,nnet_input,learning_rate)