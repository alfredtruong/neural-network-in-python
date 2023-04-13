# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 10:03:57 2023

@author: ahkar

https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
"""

# imports
from random import random
from typing import Dict,List,Union
from math import exp

# types
NNET_NODE   = Dict[str,float]
NNET_LAYER  = List[NNET_NODE]
NNET        = List[NNET_LAYER]

ATTRS       = List[float]
CLASS       = int
NNET_INPUT  = Union[ATTRS,CLASS]

CLASS_1HOT  = List[int] # 1-hot encoding of CLASS, transform int (class) into List[int] (class vector)
NNET_OUTPUT = CLASS_1HOT # 1-hot encoding prediction from nnet into class vector

DATASET     = List[List[NNET_INPUT]]

# initialize node
def initialize_node(n_node_inputs : int) -> NNET_NODE :
    return {
        'bias'    : random(),
        'weights' : [random() for i in range(n_node_inputs)]
    }

# initialize layer
def initialize_layer(n_layer_inputs : int, n_layer_nodes : int) -> NNET_LAYER :
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
    return 1 / (1 + exp(-activation))

# forward propagate input through network
def forward_propagate(nnet : NNET, nnet_input : NNET_INPUT, verbose : int = 0) -> NNET_OUTPUT:
    layer_inputs = nnet_input
    for nnet_layer in nnet:
        if verbose>0: print(f'inputs \t\t: {layer_inputs}')
        layer_outputs = []
        for layer_node in nnet_layer:
            activation = activate(layer_node['bias'],layer_node['weights'],layer_inputs)
            layer_node['output'] = transfer(activation) # enrich node dico with 'output'
            layer_outputs.append(layer_node['output'])
        if verbose>0: print(f'outputs \t: {layer_outputs}')
        if verbose>0: print()
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
            attrs = nnet_input[:-1]
        else:
            # is output from previous layer for non-input layers
            attrs = [node['output'] for node in nnet[i-1]]
        # update weights for all nodes in layer
        for node in nnet[i]:
            # update node bias
            node['bias'] -= learning_rate * node['delta']
            # update node input weighting
            for j in range(len(attrs)):
                node['weights'][j] -= learning_rate * node['delta'] * attrs[j]
    
def class_to_1hot(n_outputs : int, class_idx : CLASS) -> CLASS_1HOT :
    onehot = [0 for i in range(n_outputs)]
    onehot[class_idx] = 1
    return onehot
    
# train a network for a fixed number of epochs
def train_nnet(nnet : NNET, training_data : DATASET, learning_rate : float, n_epoch : int, n_outputs : int):
    for epoch in range(n_epoch):
        sum_error = 0
        for nnet_input in training_data:
            # compute nnet error given in put
            nnet_output = forward_propagate(nnet,nnet_input)
            expected = class_to_1hot(n_outputs,nnet_input[-1])
            sum_error += sum((expected[i] - nnet_output[i])**2 for i in range(len(expected))) # increment sum_error
            # back propagate error through nnet
            back_propagate_error(nnet,expected)
            # update nnet weights
            update_weights(nnet,nnet_input,learning_rate)
            
        if epoch % 50 == 0:
            print(f'>epoch={epoch}, learning_rate={learning_rate}, error={sum_error}')

'''
# Test training backprop algorithm
seed(1)
dataset = [
    [2.7810836,2.550537003,0],
    [1.465489372,2.362125076,0],
    [3.396561688,4.400293529,0],
    [1.38807019,1.850220317,0],
    [3.06407232,3.005305973,0],
    [7.627531214,2.759262235,1],
    [5.332441248,2.088626775,1],
    [6.922596716,1.77106367,1],
    [8.675418651,-0.242068655,1],
    [7.673756466,3.508563011,1]
]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
nnet = initialize_nnet(n_inputs, 2, n_outputs)
train_nnet(nnet, dataset, 0.5, 20, n_outputs)
print_nnet(nnet)
'''

# make a prediction with a network
def predict(nnet : NNET,nnet_input : NNET_INPUT) -> CLASS:
    nnet_output = forward_propagate(nnet,nnet_input)
    return nnet_output.index(max(nnet_output))

'''
for row in dataset:
    prediction = predict(nnet,row)
    print(f'expected={row[-1]}, got={prediction}')
'''

# back-propagation algorithm with stochastic gradient descent
def back_propagation(
    training_set : DATASET,
    test_set : DATASET,
    learning_rate : float,
    n_epoch : int,
    n_hidden : int,
    ) -> List[float] :
    # setup nnet
    n_inputs = len(training_set[0]) - 1
    n_outputs = len(set([row[-1] for row in training_set]))
    nnet = initialize_nnet(n_inputs,n_hidden,n_outputs)
    
    # fit nnet
    train_nnet(
        nnet,
        training_set,
        learning_rate,
        n_epoch,
        n_outputs)
    
    # do out-of-sample predictions
    predictions = [predict(nnet,row) for row in test_set]

    # return
    return predictions
    