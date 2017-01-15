__author__ = 'vino'

import math
import string
import pickle
import random

random.seed(0)

#return a float number from a to b
def rand(a,b):
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


class NN:
    def __init__(self,ni,a,no):
        self.ni = ni + 1
        self.nh = int((ni+no)/2.0) + a
        self.no = no

        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        self.wi = makeMatrix(self.ni,self.nh)
        self.wo = makeMatrix(self.nh,self.no)

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.05,0.05)
        for i in range(self.nh):
            for j in range(self.no):
                self.wo[i][j] = rand(-0.05,0.05)
        self.ci = makeMatrix(self.ni,self.nh)
        self.co = makeMatrix(self.nh,self.no)


    def update(self,inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')
        for i in range(self.ni-1):
            self.ah[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
        self.ah[j] = tanh(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = tanh(sum)

        return self.ao[:]


    def backPropagate(self,targets,N,M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')
        output_detas = [0.0] * self.no

        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_detas[k] = derivative_tanh(self.ao[k]) * error

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_detas[k] * self.wo[j][k]
            hidden_deltas[j] = derivative_tanh(self.ah[j]) * error

        for j in range(self.nh):
            for k in range(self.no):
                change = output_detas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]

        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2

        return error


    def test(self,patterns):
        pre = []
        for p in patterns:
            pre.append(self.update(p[0]))
            print(p,'->',self.update(p[0]))
        return pre


    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])


    def train(self, patterns, iterations=1000, N=0.01, M=0.01):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if error < 0.00001:
                    break


def demo():
    # Teach network XOR function
    pkl_file = open('train.pkl','rb')  #data_Q1.pkl
    pat = pickle.load(pkl_file)
    pkl_file = open('test.pkl','rb')  #data_Q1.pkl
    pat1 = pickle.load(pkl_file)
    #print pat,pat1

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 4, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    pre = n.test(pat1)

if __name__ == '__main__':
    demo()
