# uncompyle6 version 3.5.0
# Python bytecode 3.8 (3413)
# Decompiled from: Python 2.7.5 (default, Nov 16 2020, 22:23:17) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-44)]
# Embedded file name: layers_v3.py
# Size of source mod 2**32: 9912 bytes
import numpy as np, math
from abc import ABC, abstractmethod

class Layer(ABC):

    def __init__(self):
        self._Layer__prevIn = []
        self._Layer__prevOut = []

    def setPrevIn(self, dataIn):
        self._Layer__prevIn = dataIn

    def setPrevOut(self, out):
        self._Layer__prevOut = out

    def getPrevIn(self):
        return self._Layer__prevIn

    def getPrevOut(self):
        return self._Layer__prevOut

    def backward(self, gradIn):
        sg = self.gradient()
        return gradIn * sg

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass


class InputLayer(Layer):

    def __init__(self, dataIn, zscore=1):
        super().__init__()
        self._InputLayer__zscore = zscore
        self._InputLayer__meanX = np.mean(dataIn, axis=0)
        self._InputLayer__stdX = np.std(dataIn, axis=0, ddof=1)
        self._InputLayer__stdX[self._InputLayer__stdX == 0] = 1

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        if self._InputLayer__zscore == 1:
            temp = (dataIn - self._InputLayer__meanX) / self._InputLayer__stdX
        else:
            temp = dataIn
        self.setPrevOut(temp)
        return temp

    def gradient(self):
        pass


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
        reg = 0.0
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


class ConvolutionalLayer(Layer):

    def __init__(self, kernelSize=3, numKernels=1):
        super().__init__()
        self._ConvolutionalLayer__weights = math.pow(10.0, -4) * (np.random.rand(kernelSize, kernelSize, numKernels) - 0.5)

    def setKernels(self, kernels):
        self._ConvolutionalLayer__weights = kernels

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        temp = np.zeros((dataIn.shape[0], dataIn.shape[1] - self._ConvolutionalLayer__weights.shape[0] + 1, dataIn.shape[2] - self._ConvolutionalLayer__weights.shape[0] + 1, self._ConvolutionalLayer__weights.shape[2]))
        for n in range(dataIn.shape[0]):
            for i in range(self._ConvolutionalLayer__weights.shape[2]):
                temp[n, :, :, i] = signal.correlate2d(dataIn[n, :, :].T, self._ConvolutionalLayer__weights[:, :, i], 'valid').T

        self.setPrevOut(temp)
        return temp

    def backward(self, gradIn):
        sumGradJK = np.zeros(self._ConvolutionalLayer__weights.shape)
        prevIn = self.getPrevIn()
        for n in range(gradIn.shape[0]):
            gradJK = np.zeros(self._ConvolutionalLayer__weights.shape)
            for k in range(gradIn.shape[3]):
                for r in range(self._ConvolutionalLayer__weights.shape[0]):
                    for c in range(self._ConvolutionalLayer__weights.shape[1]):
                        dFdKij = prevIn[n, r:r + prevIn.shape[1] - self._ConvolutionalLayer__weights.shape[0] + 1, c:c + prevIn.shape[2] - self._ConvolutionalLayer__weights.shape[1] + 1]
                        gradJK[(r, c, k)] = signal.correlate2d(gradIn[n, :, :, k], dFdKij, 'valid')

            sumGradJK += gradJK

        self._ConvolutionalLayer__weights = self._ConvolutionalLayer__weights - eta / gradIn.shape[0] * sumGradJK

    def gradient(self):
        pass


class MaxPoolLayer(Layer):

    def __init__(self, size=3, stride=1):
        super().__init__()
        self._MaxPoolLayer__size = size
        self._MaxPoolLayer__stride = stride


    def gradient(self):
        pass


class FlatteningLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        size = dataIn.shape
        temp = np.reshape(dataIn, (size[0], size[1] * size[2] * size[3]))
        self.setPrevOut(temp)
        return temp

    def backward(self, gradIn):
        return np.reshape(gradIn, (self.getPrevIn().shape), order='F')

    def gradient(self):
        pass


class LinearLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.copy(dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        po = self.getPrevIn()
        return np.ones(po.shape)


class ReLuLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = np.copy(dataIn)
        Y[Y < 0] = 0
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        po = self.getPrevIn()
        return 1.0 * (po >= 0)


class SigmoidLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = 1 / (1 + np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        eps = 1e-07
        po = self.getPrevOut()
        return po * (1 - po) + eps


class TanhLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        po = self.getPrevOut()
        return 1 - po * po


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        m = np.array([np.max(dataIn, axis=1)]).T
        tmp = np.exp(dataIn - m)
        sm = np.array([np.sum(tmp, axis=1)]).T
        sm = tmp / sm
        self.setPrevOut(sm)
        return sm

    def gradient(self):
        po = self.getPrevOut()
        grad = np.zeros([po.shape[0], po.shape[1], po.shape[1]])
        for i in range(po.shape[0]):
            s = po[i, :]
            s1 = s.reshape(-1, 1)
            grad[i, :, :] = np.diagflat(s) - np.dot(s1, s1.T)

        return grad

    def gradient2(self):
        pi = self.getPrevIn()
        v = self.getPrevOut()
        grad = np.zeros([pi.shape[0], pi.shape[1], pi.shape[1]])
        for i in range(pi.shape[1]):
            for j in range(pi.shape[1]):
                if i == j:
                    grad[:, i, j] = v[:, i] * (1 - v[:, i])
                else:
                    grad[:, i, j] = -v[:, i] * v[:, j]

        return grad

    def backward(self, gradIn):
        sg = self.gradient()
        grad = np.zeros((gradIn.shape[0], sg.shape[2]))
        for n in range(gradIn.shape[0]):
            grad[n, :] = gradIn[n, :] @ sg[n, :, :]

        return grad


class LeastSquares:

    def eval(self, y, yhat):
        return np.mean((y - yhat).T @ (y - yhat))

    def gradient(self, y, yhat):
        tmp = -2 * (y - yhat)
        return tmp


class LogLoss:

    def eval(self, y, yhat):
        eps = 1e-07
        return -np.mean(y * np.log(yhat + eps) + (1 - y) * np.log(1 - yhat + eps))

    def gradient(self, y, yhat):
        eps = 1e-07
        tmp = (1 - y) / (1 - yhat + eps) - y / (yhat + eps)
        return tmp


class GANLoss:

    def eval(self, y, yhat):
        eps = 1e-07
        return -np.mean(np.log(yhat + eps))

    def gradient(self, y, yhat):
        eps = 1e-07
        tmp = -1.0 / (yhat + eps)
        return tmp


class CrossEntropy:

    def eval(self, y, yhat):
        eps = 1e-07
        tmp = y * np.log(yhat + eps)
        tmp[y == 0] = 0
        val = -np.mean(tmp)
        return val

    def gradient(self, y, yhat):
        eps = 1e-07
        tmp = -y / (yhat + eps)
        tmp[y == 0] = 0
        return tmp


if __name__ == '__main__':
    x = InputLayer()
