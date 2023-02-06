# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:46:06 2023

@author: ahkar
"""

from nnet import nnet # https://stackoverflow.com/questions/2349991/how-do-i-import-other-python-files
from nnet import data # https://stackoverflow.com/questions/2349991/how-do-i-import-other-python-files
from random import seed

dir(nnet)
dir(data)

##################################
# load and prepare data
##################################
filename = 'wheat-seeds.txt'
dataset = data.load_csv(filename)

# fix column data types, attrs to float
for i in range(len(dataset[0])-1):
    data.str_column_to_float(dataset,i)
    
# fix column data types, class to integer 
data.str_column_to_int(dataset,len(dataset[0])-1)

# normalize input variables
minmax = data.dataset_minmax(dataset)
data.normalize_dataset(dataset,minmax)

##################################
# evaluate algorithm
##################################
seed(1)

n_folds = 5
learning_rate = 0.3
n_epoch = 500
n_hidden = 5

scores = data.evaluate_algorithm(
    # named
    dataset,
    nnet.back_propagation,
    n_folds,
    # *args
    learning_rate,
    n_epoch,
    n_hidden,
)

