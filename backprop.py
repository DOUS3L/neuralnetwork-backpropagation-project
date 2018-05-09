import numpy as np

class Network(object):
    
    def __init__(self, sizes):
        # sizes = [2,3,1] means 2 neuron on input layer, 3 neuron on hidden layer
        # and 1 on the 3rd layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        
        for i in range(0,len(self.biases)):
            self.biases[i] = np.squeeze(self.biases[i])
            
    def train(self, X_train, y_train, epochs, eta, error_threshold):
        self.eta = eta
        self.totalerrors = [1]
        self.errors = []
        self.outs = [np.zeros(y) for y in self.sizes[1:]]
        self.deltas = [np.zeros(y) for y in self.sizes[1:]]
        self.dw = [np.zeros([self.sizes[1],self.sizes[0]]),np.zeros([self.sizes[2],self.sizes[1]])]
        self.db = [np.zeros(y) for y in self.sizes[1:]]
        epochcount = 0
        
        while True:
            for data, target in zip(X_train, y_train):
                self.feedforward(data)
                self.computeerrors(target)
                self.backpropagation(data,target)
                print(self.weights[0])
            print('epoch '+str(epochcount))
            if epochcount >= epochs:
                break
            if self.totalerrors[-1] < error_threshold:
                print(self.totalerrors[-1])
                break
            
            
            epochcount += 1
            
    def backpropagation(self, data, target):
        self.deltas[1] = (target-self.outs[1]) * self.dsigmoid(self.outs[1])
        self.dw[1] = np.array([[x*y for y in self.outs[0]] for x in self.deltas[1]]) * self.eta
        self.db[1] = self.deltas[1] * self.eta

        self.deltas[0] = self.deltas[1].dot(self.weights[1]) * self.dsigmoid(self.outs[0])
        self.dw[0] = np.array([[x*y for y in data]for x in self.deltas[0]])
        self.db[0] = self.deltas[0] * self.eta
        
        self.weights[0] = self.weights[0] + self.dw[0]
        self.biases[0] = self.biases[0] + self.db[0]

        self.weights[1] = self.weights[1] + self.dw[1]
        self.biases[1] = self.biases[1] + self.db[1]
        


    def feedforward(self, data):
        self.outs[0] = self.sigmoid(np.dot(self.weights[0], data) + self.biases[0])
        self.outs[1] = self.sigmoid(np.dot(self.weights[1], self.outs[0]) + self.biases[1])

    def computeerrors(self, target):
        self.errors.append(target-self.outs[-1])
        self.totalerrors.append(np.max(np.abs(self.errors[-1])))     
        
    #def computeerrors(self, target):
    #    self.errors.append(0.5 * (target-self.outs[-1])**2)
    #    self.totalerrors.append(sum(self.errors[-1]))     
        
    def sigmoid(self, x):
        return (2.0 / (1.0 + np.exp(-x)))-1

    def dsigmoid(self, x):
        return 0.5 * (1+self.sigmoid(x)) * (1-self.sigmoid(x))
    
    def test(self, data, target):        
        out0 = self.sigmoid(np.dot(self.weights[0], data) + self.biases[0])
        out1 = self.sigmoid(np.dot(self.weights[1], out0) + self.biases[1])
        error = np.max(np.abs(target-out1))
        print('{} {} {}'.format(out1, target, error))
        status = True
        if error > 1:
            status = False       
        
        
        
        #error = sum(0.5 * (target-out1)**2)
        return [out1, target,error, status]
        
        
        
        
        
        
    









