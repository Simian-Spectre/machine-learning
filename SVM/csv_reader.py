import numpy as np
import csv

# Takes in a path to a csv file and the number of features
# Assumes the label is placed after the features
# For example, a csv with three features and a label in the 
# fourth position should pass in 3 as the num_features param
# Returns two np arrays:
#   examples: an np array of the examples, each example is a list of features
#   labels: an np array of the labels (0 or 1) associated with each example
def np_arrays_from_csv(path_to_csv, num_features):
    examples = []
    labels = []
    with open (path_to_csv, 'r') as csv:
        for line in csv:
            line_as_list = line.strip().split(',')
            examples.append(line_as_list[:num_features])
            labels.append(line_as_list[num_features])

    return np.array(examples, dtype=float), np.array(labels, dtype=float)