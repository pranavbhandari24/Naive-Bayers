# Name      :   Pranav Bhandari
# Student ID:   1001551132
# Date      :   09/15/2020


import numpy as np
import sys, math

class Data_Class:
    def __init__(self, name, num_features, prior_probability):
        self.classname = name
        self.num_features = num_features
        self.training_data = [[] for i in range(num_features)]
        self.stdev = []
        self.mean = []
        self.prior_probability = prior_probability

    def calculate_stats(self):
        for i in range(self.num_features):
            self.mean.append(np.average(self.training_data[i]))
            stdev = np.std(self.training_data[i])
            if stdev == 0.0:
                self.stdev.append(0.01)
            else:
                self.stdev.append(stdev)

    def print_stats(self):
        for i in range(self.num_features):
            print("Class {}, attribute {}, mean = {:.2f}, std = {:.2f}".format(self.classname, i+1, self.mean[i], self.stdev[i]))

def calculate_gaussian(mean, var, x):
    epsilon = 1e-4 
    coeff = 1.0 / math.sqrt(2.0 * np.pi * var + epsilon)
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + epsilon)))
    return coeff * exponent

def training(training_file):
    file = open(training_file, "r")
    numbers = [[float(x) for x in line.split()] for line in file]
    numbers_array = np.array(numbers)
    columns = len(numbers[0])
    rows = len(numbers)

    # Number of attributes 
    num_features = columns-1
    # Number of Unique Classes
    unique_classnames, counts = np.unique(numbers_array[:,columns-1], return_counts=True)
    num_classes = len(unique_classnames)

    classes = []
    for i in range(len(unique_classnames)):
        prior_probability = float(counts[i])/rows
        c = Data_Class(int(unique_classnames[i]), num_features, prior_probability)
        classes.append(c)

    for training_data in range (rows):
        curr_class = numbers[training_data][columns-1]
        index = (np.where(unique_classnames == curr_class))[0][0]
        c = classes[index]
        for feature in range (columns-1):
            c.training_data[feature].append(numbers[training_data][feature])
    
    for i in range(num_classes):
        classes[i].calculate_stats()
        classes[i].print_stats()
    
    return classes

def classify(classes, test_data):
    num_classes = len(classes)
    num_features = classes[0].num_features
    p_x_given_class = []
    p_x = 0
    for i in range(num_classes):
        probability = 1
        for feature in range(num_features):
            stdev = classes[i].stdev[feature]
            var = math.pow(stdev, 2)
            mean = classes[i].mean[feature]
            x = test_data[feature]
            probability *= calculate_gaussian(mean, var, x)
        p_x_given_class.append(probability)
        p_x += (probability * classes[i].prior_probability)
    max_prob = -1.0
    result = -1
    for i in range(num_classes):
        p_class_given_x = (p_x_given_class[i] * classes[i].prior_probability) / p_x
        if p_class_given_x > max_prob:
            result = classes[i].classname
            max_prob = p_class_given_x
    return result, max_prob

def testing(classes, test_file):
    file = open(test_file, "r")
    numbers = [[float(x) for x in line.split()] for line in file]
    rows = len(numbers)
    columns = len(numbers[0])
    num_features = columns-1
    num_correct = 0

    for row in range(rows):
        name, probability = classify(classes, numbers[row])
        true_value = int(numbers[row][columns-1])
        if name == true_value:
            accuracy = 1.0 
            num_correct +=1
        else:
            accuracy = 0.0
        print("ID={:5d}, predicted={:3d}, probability = {:.4f}, true={:3d}, accuracy={:4.2f}".format(row+1, name, probability, true_value, accuracy))
    
    print("classification accuracy={:.4f}".format(float(num_correct)/rows))

def naive_bayers(training_file, test_file):
    classes = training(training_file)
    testing(classes, test_file)

if __name__ == '__main__':
    naive_bayers(sys.argv[1], sys.argv[2])