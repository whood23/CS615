from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import OneHotEncoder as ohe
# import time as dt
import math


class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, dataOut):
        self.__prevOut = dataOut

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    def backward(self, gradIn):
        sg = self.gradient()
        grad = np.zeros((gradIn.shape[0], sg.shape[2]))
        for n in range(gradIn.shape[0]):
            grad[n, :] = np.matmul(gradIn[n, :], sg[n, :, :])

        return grad

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass


class InputLayer(Layer):
    def __init__(self, dataIn):
        self.dataIn = dataIn
        self.mean_data = np.mean(dataIn, axis=0)
        self.std_data = np.std(dataIn, axis=0, ddof=1)
        self.std_data[self.std_data == 0] = 1

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        z_scored_data = (dataIn - self.mean_data) / self.std_data
        self.setPrevOut(z_scored_data)
        return z_scored_data

    def gradient(self):
        pass

    def backward(self, gradIn):
        pass



# Objective Functions
class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        linear_data = np.copy(dataIn)
        self.setPrevOut(linear_data)
        return linear_data

    def gradient(self):
        dataIn = np.copy(self.getPrevIn())
        totRows = np.size(dataIn, axis=0)  # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1)  # Grab the total number of columns
        tensor = np.random.rand(0, totColumns, totColumns)  # Create tensor
        # Preform Gradient calculation
        dataIn[dataIn >= 0] = 1
        dataIn[dataIn < 0] = 1
        # Create final matrix and fill tensor
        for row in range(totRows):
            gradData = np.zeros((totColumns, totColumns))
            np.fill_diagonal(gradData, dataIn[row])
            tensor = np.concatenate((tensor, gradData[None]), axis=0)

        return tensor


class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        relu_data = np.copy(dataIn)
        relu_data[relu_data < 0] = 0
        self.setPrevOut(relu_data)
        return relu_data

    def gradient(self):
        dataIn = np.copy(self.getPrevIn())
        totRows = np.size(dataIn, axis=0)  # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1)  # Grab the total number of columns
        tensor = np.random.rand(0, totColumns, totColumns)  # Create tensor
        # Preform Gradient calculation
        dataIn[dataIn >= 0] = 1
        dataIn[dataIn < 0] = 0
        # Create final matrix and fill tensor
        for row in range(totRows):
            gradData = np.zeros((totColumns, totColumns))
            np.fill_diagonal(gradData, dataIn[row])
            tensor = np.concatenate((tensor, gradData[None]), axis=0)

        return tensor


class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        sig_data = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(sig_data)
        return sig_data

    def gradient(self):
        dataIn = self.getPrevOut()
        totRows = np.size(dataIn, axis=0)  # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1)  # Grab the total number of columns
        tensor = np.random.rand(0, totColumns, totColumns)  # Create an empty tensor
        # Create Preform calculation, final matrix, and tensor append 
        for row in range(totRows):
            diagonal = dataIn[row] * (1 - dataIn[row])  # diagonal calculation
            gradData = np.zeros((totColumns, totColumns))
            np.fill_diagonal(gradData, diagonal)
            tensor = np.concatenate((tensor, gradData[None]), axis=0)
        return tensor


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        m = np.array([np.max(dataIn, axis=1)]).T
        tmp = np.exp(dataIn - m)
        sm = np.array([np.sum(tmp, axis=1)]).T
        sm = tmp/sm
        self.setPrevOut(sm)
        return sm

    def gradient(self):
        dataIn = self.getPrevOut()
        totRows = np.size(dataIn, axis=0)  # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1)  # Grab the total number of columns
        tensor = np.random.rand(0, totColumns, totColumns)  # Create an empty tensor
        # Run calculation per row, add expanded matrix to empty tensor
        for row in range(totRows):
            gradData = np.zeros((totColumns, totColumns))
            aIndex = 0
            while aIndex < totColumns:
                for bIndex in range(0, len(gradData)):
                    if bIndex == aIndex:
                        gradData[bIndex, aIndex] = dataIn[row, aIndex] * (1 - dataIn[row, aIndex])
                    else:
                        gradData[bIndex, aIndex] = -dataIn[row, aIndex] * dataIn[row, bIndex]
                aIndex += 1

            tensor = np.concatenate((tensor, gradData[None]), axis=0)

        return tensor


class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        tanh_data = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(tanh_data)
        return tanh_data

    def gradient(self):
        dataIn = self.getPrevOut()
        totRows = np.size(dataIn, axis=0)  # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1)  # Grab the total number of columns
        tensor = np.random.rand(0, totColumns, totColumns)  # Create an empty tensor
        # Create Preform calculation, final matrix, and tensor append
        for row in range(totRows):
            diagonal = 1 - dataIn[row] ** 2  # diagonal calculation
            gradData = np.zeros((totColumns, totColumns))
            np.fill_diagonal(gradData, diagonal)
            tensor = np.concatenate((tensor, gradData[None]), axis=0)

        return tensor


# Fully Connected Without Adam
class FullyConnected2(Layer):

    def __init__(self, sizeIn, sizeOut):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        self.weights = np.random.uniform(-0.0001, 0.0001, size=(sizeIn, sizeOut))
        self.bias = np.random.uniform(-0.0001, 0.0001, size=(1, sizeOut))

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self, bias):
        self.bias = bias

    def updateWeights(self, gradIn, eta=0.0001):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        dJdw = (np.matmul(self.getPrevIn().T, gradIn) / gradIn.shape[0])
        self.bias = self.bias - eta * dJdb
        self.weights = self.weights - (eta * dJdw)

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        newSet = np.matmul(dataIn, self.weights) + self.bias
        self.setPrevOut(newSet)
        return newSet

    def gradient(self):
        weights = self.getWeights()
        grad_fc_data = np.transpose(weights)
        return grad_fc_data

    def backward(self, gradIn):
        sg = self.gradient()
        grad = np.matmul(gradIn, sg)
        return grad


# Fully Connected with adam
class FullyConnected(Layer):

    def __init__(self, sizein, sizeout, useBias=1):
        super().__init__()
        self._FullyConnected__weights = 0.0001 * (np.random.rand(sizein, sizeout) - 0.5)
        self._FullyConnected__biases = 0.0001 * (np.random.rand(1, sizeout) - 0.5)
        self._FullyConnected__sW = 0
        self._FullyConnected__rW = 0
        self._FullyConnected__sb = 0
        self._FullyConnected__rb = 0
        self._FullyConnected__useBias = useBias

        

    def setWeights(self, weights):
        self._FullyConnected__weights = weights

    def getWeights(self):
        return self._FullyConnected__weights

    def setBias(self, bias):
        self._FullyConnected__biases = bias

    def getBias(self):
        return self._FullyConnected__bias

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        if self._FullyConnected__useBias == 1:
            temp = dataIn @ self._FullyConnected__weights + self._FullyConnected__biases
        else:
            temp = dataIn @ self._FullyConnected__weights
        self.setPrevOut(temp)
        return temp

    def backward(self, gradIn):
        return gradIn @ self._FullyConnected__weights.T

    def updateWeights(self, gradIn, eta, p1=0.9, p2=0.999, rho=-14, epoch=-1):
        reg = 0
        pi = self.getPrevIn()
        po = self.getPrevOut()
        dJdW = pi.T @ gradIn
        deltaJ = dJdW / gradIn.shape[0] + 2 * reg * self._FullyConnected__weights / gradIn.shape[0]
        self._FullyConnected__sW = p1 * self._FullyConnected__sW + (1 - p1) * deltaJ
        self._FullyConnected__rW = p2 * self._FullyConnected__rW + (1 - p2) * (deltaJ * deltaJ)
        dJdb = np.sum(gradIn, 0)
        deltaJ = dJdb / gradIn.shape[0] + 2 * reg * self._FullyConnected__biases / gradIn.shape[0]
        self._FullyConnected__sb = p1 * self._FullyConnected__sb + (1 - p1) * deltaJ
        self._FullyConnected__rb = p2 * self._FullyConnected__rb + (1 - p2) * (deltaJ * deltaJ)
        if epoch == -1:
            self._FullyConnected__weights -= eta * dJdW / pi.shape[0]
            self._FullyConnected__biases -= eta * dJdb / pi.shape[0]
        else:
            self._FullyConnected__weights -= eta * (self._FullyConnected__sW / (1 - p1 ** epoch) / (np.sqrt(self._FullyConnected__rW / (1 - p2 ** epoch)) + rho) + 2 * reg * self._FullyConnected__weights / gradIn.shape[0])
            self._FullyConnected__biases -= eta * (self._FullyConnected__sb / (1 - p1 ** epoch) / (np.sqrt(self._FullyConnected__rb / (1 - p2 ** epoch)) + rho) + 2 * reg * self._FullyConnected__biases / gradIn.shape[0])

    def gradient(self):
        return np.tile(self._FullyConnected__weights.T, (self.getPrevIn().shape[0], 1, 1))



# Evaluation methods
class LeastSquares:
    def eval(self, y, yhat):
        j = (y - yhat) ** 2
        return j

    def gradient(self, y, yhat):
        grad_j = -2 * (y - yhat)
        return grad_j

    def backward(self, gradIn):
        pass


class LogLoss:
    def eval(self, y, yhat):
        eps = 1e-07
        return -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))

    def gradient(self, y, yhat):
        eps = 1e-07
        tmp = (1 - y) / (1 - yhat + eps) - y / (yhat + eps)
        return tmp

    def backward(self, gradIn):
        pass


class CrossEntropy:
    def eval(self, y, yhat):
        eps = 1e-07
        j = y * np.log(yhat + eps)
        val = -np.mean(j)
        return val

    def gradient(self, y, yhat):
        grad_j = -y / (yhat + 1e-7)
        return grad_j

    def backward(self, gradIn):
        pass



