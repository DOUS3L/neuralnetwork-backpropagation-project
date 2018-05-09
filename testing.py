import numpy as np
import pandas as pd
from backprop import Network  
    #(self, X_train, y_train, epochs, eta, error_threshold):


dataset = pd.read_csv('data.csv')

data = np.array(dataset)
X = data[:,0:12]
y = data[:,13:]

testing = pd.read_csv('testing.csv')

datest = np.array(testing)
X_test = datest[:,0:12]
y_test = datest[:,13:]


nn = Network([12,4,2])
nn.train(X,y, 1000, 0.5, 0.001)

result = []
for i,k in zip(X_test, y_test):
    result.append(list(nn.test(i,k)))

persen = 100-(len([x for x in result if x[3] is False])/len(result))*100








