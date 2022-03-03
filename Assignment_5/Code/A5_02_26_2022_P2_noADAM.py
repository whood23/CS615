from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
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
        self.std_data = np.std(dataIn, axis=0)
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


class FullyConnectedLayer(Layer):
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
        self.bias = self.bias - (eta * dJdb)
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
        j = -1 * ((np.log(yhat + 10e-7) * y) + (1 - y) * np.log(1 - yhat + 10e-7))
        return j

    def gradient(self, y, yhat):
        grad_j = -(y - yhat) / (yhat * (1 - yhat) + 10e-7)
        return grad_j

    def backward(self, gradIn):
        pass


class CrossEntropy:
    def eval(self, y, yhat):
        eps = 1e-07
        #yhat[yhat<=0] = 10e-7
        #yhet = np.log(yhat.T)
        j = y * np.log(yhat + eps)
        #j = np.dot(-y, np.log(np.transpose(yhat)))
        #j = np.matmul(yhet, -y)
        j[y==0]=0
        val = -np.mean(j)
        return val

    def gradient(self, y, yhat):
        grad_j = -y / (yhat + 10e-7)
        return grad_j

    def backward(self, gradIn):
        pass


class RunLayers:
    def __init__(self, X, Y, layerList, epochs, eta, eval_method='none'):
        self.X = X
        self.Y = Y
        self.layers = layerList
        self.epochs = epochs
        self.eta = eta
        self.eval_method = eval_method

    def forwardRun(self, X):
        H = X
        for i in range(len(self.layers) - 1):
            H = self.layers[i].forward(H)
            print(self.layers[i], H)
        return H

    def backRun(self, Y, H):
        grad = self.layers[-1].gradient(Y, H)
        for i in range(len(self.layers) - 2, 0, -1):
            newGrad = self.layers[i].backward(grad)

            if isinstance(self.layers[i], FullyConnectedLayer):
                self.layers[i].updateWeights(grad, self.eta)

            grad = newGrad

    def mapeRun(self, H):
        mape = np.mean(np.absolute((self.Y - H) / self.Y))
        return mape

    def rmseRun(self, H):
        rmse = math.sqrt(np.matmul(np.transpose(self.Y - H), (self.Y - H)) / np.size(H, axis=0))
        return rmse

    def objfunRun(self, H):
        objRun = np.mean(self.layers[-1].eval(self.Y, H))
        return objRun

    def objSelect(self, H):
        if self.eval_method == 'none':
            return self.objfunRun(H)
        elif self.eval_method == 'mape':
            return self.mapeRun(H)
        elif self.eval_method == 'rmse':
            return self.rmseRun(H)

    def allRun(self):
        endDiff = 1e-10
        epochStorage = []
        errorStorage = []
        prevError = 0

        for j in range(self.epochs):
            # Forward
            H = self.forwardRun(self.X)
            # Backwards
            self.backRun(self.Y, H)
            error = self.objSelect(H)
            errorStorage.append(error)
            epochStorage.append(j)

            if np.absolute(error - prevError) < endDiff:
                return epochStorage, errorStorage
                break

            prevError = error

        return epochStorage, errorStorage

    def classify(self, X):
        classification = self.forwardRun(X)
        classification[classification < 0.5] = 0
        classification[classification >= 0.5] = 1
        return classification

if __name__ == '__main__':
    start_time = dt.now()
    # start_time = dt.time()
    def plot(xPlotData, yPlotData):
        plt.plot(xPlotData, yPlotData)
        plt.xlabel('Epoch')
        plt.ylabel('Evaluation Function')
        return plt


    data = np.genfromtxt('mnist_train_100.csv', delimiter=',')
    
    XTrain = data[:, 1:]
    YTrain = data[:, :1]

    L1 = InputLayer(XTrain)
    L2 = FullyConnectedLayer(XTrain.shape[1], 10)
    L3 = SigmoidLayer()
    L4 = LogLoss()
    layers = [L1, L2, L3, L4]
    ep = 1000
    # print("Number of epochs: {}".format(ep))
    "Training"
    # Run test
    run = RunLayers(XTrain, YTrain, layers, ep, 0.5)
    epochStorageTrain, errorStorageTrain = run.allRun()
    trainClassify = run.classify(XTrain)
    binaryClassify = (YTrain == trainClassify)
    

    print("Training Accuracy: {0:.2f}%".format((np.count_nonzero(binaryClassify) / np.size(YTrain, axis=0)) * 100))
    
    end_time = dt.now()
    print("Duration: {}".format(end_time - start_time))
    plt.figure(3)
    plt.plot(epochStorageTrain, errorStorageTrain)
    plt.show()
 
