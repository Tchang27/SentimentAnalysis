import numpy as np
from tqdm import tqdm

class NeuralNet:
    def __init__(self, layers, alpha = 0.1):
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i]+1, layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNet: {}".format(
            "-".join(str(l)for l in self.layers)
        )

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    def _sigmoid_deriv(self, x):
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=50):
        '''
        Param:
        X - training set
        y - labels for training set
        epochs - iterations of training
        displayUpdate - intervals to display progress
        '''
        X = np.c_[X, np.ones((X.shape[0]))]
        counter = 0
        for epoch in np.arange(0, epochs):
            for x, target in zip(X, y):
                self.fit_partial(x, target)
  
            if epoch == 0 or (epoch+1)%displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.7f}".format(
                epoch + 1, loss))
            
            counter += 1
            print("epoch " + str(counter) + " complete")

    def fit_partial(self, x, y):
        #forwward propagation
        A = [np.atleast_2d(x)]
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)

        #calculating deltas
        error = A[-1] - y
        D = [error*self._sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A)-2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self._sigmoid_deriv(A[layer])
            D.append(delta)
        D = D[::-1]

        #updating deltas:
        for layer in np.arange(0, len(self.W)):
            #update weights by taking dot product of layer activations
            #with deltas, then multiplying by activation rate
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        '''
        Param:
        X - test data
        addBias - if bias is wanted
        '''
        p = np.atleast_2d(X)
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p
    
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predicitions = self.predict(X, addBias=False)
        to_sum = np.multiply((predicitions-targets), (predicitions-targets))
        loss = 0.5*np.sum(to_sum)
        return loss


