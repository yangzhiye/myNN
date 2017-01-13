__author__ = 'vino'

import math
import string
import pickle
import random

random.seed(0)

#return a float number from a to b
def return_rand(a,b):
    return random.uniform(a,b)

#return a Matrix filled by fill , I is row , J is column
def makeMatrix(I,J,fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

#default activation function is tanh
def tanh(x):
    return math.tanh(x)

# tanh's derivative
def derivative_tanh(y):
    return 1.0 - y**2



if __name__ == "__main__":
    