# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:06:06 2023

@author: ahkar
"""

# imports
from csv import reader
from typing import List
from typing import Dict
from random import randrange

# types
CLASS = int

# load csv file
def load_csv(filename) -> List[str]:
    dataset=[]
    with open(filename,'r') as f:
        csv_reader = reader(f)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
        return dataset
    
# convert string column to float
def str_column_to_float(dataset,column) -> None:
    for row in dataset:
        row[column] = float(row[column].strip())
        
# convert string column to integer
def str_column_to_int(dataset,column) -> Dict[str,CLASS]:
    # extract unique classes and convert it into a categoric variable
    class_values=[row[column] for row in dataset]
    unique = set(class_values)
    lookup = {}
    for i,value in enumerate(unique):
        lookup[value] = i
        
    # do conversion in dataset
    for row in dataset:
        row[column] = lookup[row[column]]
        
    # return mapping
    return lookup

# find min and max values for each column
def dataset_minmax(dataset) -> List[float]:
    stats  = [(min(column),max(column)) for column in zip(*dataset)]
    return stats

# rescale dataset columns to 0-1 range
def normalize_dataset(dataset,minmax) -> None:
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
        
# split a dataset into k-folds
def cross_validation_split(dataset,n_folds) -> List[List[int]] :
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)) / n_folds
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# calculate accuracy percentage
def accuracy_metric(actual,predicted) -> float :
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100

# evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset,algorithm,n_folds,*args):
    folds = cross_validation_split(dataset,n_folds)
    scores = []
    for fold in folds:
        # training set
        training_set = list(folds)
        training_set.remove(fold)
        training_set = sum(training_set,[])
        # test set
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(training_set,test_set,*args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual,predicted)
        scores.append(accuracy)
    return scores