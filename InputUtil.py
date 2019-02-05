import os
import PatternUtil
from numpy import array, asarray

def load_training_data():
    labels = []
    training_data = []
    print("Loading training data.") 
    file = open("training_data.txt", "r")
    for line in file:
        label = line.split(",")[0].strip()
        letter = line.split(",")[1].strip()
        pattern = PatternUtil.to_nparray_bin(letter)
        labels.append(label)
        training_data.append(pattern)
        PatternUtil.display_pattern(pattern)
    return labels, asarray(training_data)

def load_test_data():
    test_data = []
    print("Loading test data from test_data.txt") 
    file = open("test_data.txt", "r")
    for line in file:
        letter = line.strip()
        pattern = PatternUtil.to_nparray_bin(letter)
        test_data.append(pattern)
    return asarray(test_data)
