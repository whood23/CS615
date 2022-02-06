# Module 1 for forward propagation in Deep Learning
# Module will contain the bones for reusable machine learning code

## Important packages
from abc import ABC, abstractmethod
import numpy as np

### Edit the forward section
class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn
    
    def setPrevOut(self,dataOut):
        self.__prevOut = dataOut

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    def backward(self,gradIn):
        pass

    @abstractmethod
    def forward(self,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

### Edit everything except the Gradient
class InputLayer(Layer):
    def __init__(self, dataIn):
        self.dataIn = dataIn
        self.mean_data = np.mean(dataIn,axis=0)
        self.std_data = np.std(dataIn,axis=0)
        self.std_data[self.std_data==0]=1

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        z_scored_data = (dataIn - self.mean_data) / self.std_data
        self.setPrevOut(z_scored_data)
        return z_scored_data

    def gradient(self):
        pass

# Activation Layers
### Edit all the Layer classes

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        linear_data = np.copy(dataIn)
        self.setPrevOut(linear_data)
        return linear_data

    def gradient(self):
        pass

class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        relu_data = np.copy(dataIn)
        relu_data[relu_data<0]=0
        self.setPrevOut(relu_data)
        return relu_data

    def gradient(self):
        dataIn = np.copy(self.getPrevIn())
        inSize = np.size(dataIn)
        dataIn[dataIn>=0]=1
        dataIn[dataIn<0]=0
        #create final array
        grad_relu_data = np.zeros((inSize,inSize))
        np.fill_diagonal(grad_relu_data,dataIn)
        return grad_relu_data

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        sig_data = 1/(1+np.exp(-dataIn))
        self.setPrevOut(sig_data)
        return sig_data

    def gradient(self):
        dataIn = self.getPrevOut()
        inSize = np.size(dataIn)
        h_diag = dataIn*(1-dataIn)
        grad_sig_data = np.zeros((inSize,inSize))
        np.fill_diagonal(grad_sig_data,h_diag)
        return grad_sig_data


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        # top, bottom = np.exp(dataIn-(np.amax(dataIn, axis=0))), np.sum(np.exp(dataIn-(np.amax(dataIn, axis=0))))
        # soft_data = top/bottom
        soft_data = np.exp(dataIn)/np.sum(np.exp(dataIn))
        self.setPrevOut(soft_data)
        return soft_data

    def gradient(self):
        dataIn = self.getPrevOut()
        inSize = np.size(dataIn)
        grad_soft_data = np.zeros((inSize,inSize))
        ar_index = 0
        while ar_index < len(dataIn[0]):
            for mat_index in range(0,len(grad_soft_data)):
                if mat_index == ar_index:
                    grad_soft_data[mat_index,ar_index] = dataIn[:,ar_index]*(1-dataIn[:,ar_index])
                else:
                    grad_soft_data[mat_index,ar_index] = -dataIn[:,ar_index] * dataIn[:,mat_index]
            ar_index+=1
        return grad_soft_data

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        tanh_data = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn)+np.exp(-dataIn))
        self.setPrevOut(tanh_data)
        return tanh_data

    def gradient(self):
        dataIn = self.getPrevOut()
        inSize = np.size(dataIn)
        h_diag = 1-dataIn**2
        grad_tanh_data = np.zeros((inSize,inSize))
        np.fill_diagonal(grad_tanh_data,h_diag)
        return grad_tanh_data


# Fully Connected Layer
### Edit all defs except the gradient

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        np.random.seed(0)
        self.weights = np.random.uniform(-0.0001, 0.0001,size=(sizeIn,sizeOut))
        self.bias = np.random.uniform(-0.0001, 0.0001,size=(1,sizeOut))

    def getWeights(self):
        return self.weights

    def setWeights(self,weights):
        self.weights = weights

    def getBias(self):
        return self.bias

    def setBias(self,bias):
        self.bias = bias

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        newset = np.matmul(dataIn,self.weights)+self.bias
        self.setPrevOut(newset)
        return newset

    def gradient(self):
        weights = self.getWeights()
        grad_fc_data = np.transpose(weights)
        return grad_fc_data

class LeastSquares:
    def eval(self,y,yhat):
        j = (y - yhat)**2
        return j

    def gradient(self,y,yhat):
        grad_j = -2*(y-yhat)
        return grad_j

class LogLoss:
    def eval(self,y,yhat):
        j = (yhat**y)*(1-yhat)**(1-y)
        return j

    def gradient(self,y,yhat):
        grad_j = -(y-yhat)/(yhat*(1-yhat))
        return grad_j

class CrossEntropy:
    def eval(self,y,yhat):
        j = np.dot(-y,np.log(np.transpose(yhat)))
        return j

    def gradient(self,y,yhat):
        grad_j = -y/yhat
        return grad_j


if __name__ == '__main__':
    test_set = np.array([[1,2,3,4]])
    print()
    print("Part 4",end = "\n")

    fc = FullyConnectedLayer(4,2)
    fc_data = fc.forward(test_set)
    print("FullyConnectedLayer Gradient")
    print(fc.gradient(), end = "\n\n")

    ru = ReLuLayer()
    ru_data = ru.forward(test_set)
    print("ReLuLayer Gradient")
    print(ru.gradient(), end = "\n\n")

    sm = SoftmaxLayer()
    sm_data = sm.forward(test_set)
    print("SoftmaxLayer Gradient")
    print(sm.gradient(), end = "\n\n")

    tanh = TanhLayer()
    tanh_data = tanh.forward(test_set)
    print("TanhLayer Gradient")
    print(tanh.gradient(), end = "\n\n")

    sig = SigmoidLayer()
    sig_data = sig.forward(test_set)
    print("SigmoidLayer Gradient")
    print(sig.gradient(), end = "\n\n")

    print("Part 5",end="\n")
    y = 0
    yhat = 0.2

    print("LeastSquares")
    ls = LeastSquares()
    print("eval:",ls.eval(y,yhat))
    print("gradient:",ls.gradient(y,yhat),end="\n\n")

    print("LogLoss")
    ls = LogLoss()
    print("eval:",ls.eval(y,yhat))
    print("gradient:",ls.gradient(y,yhat),end="\n\n")

    print("CrossEntropy")
    y = np.array([[1,0,0]])
    yhat = np.array([[0.2,0.2,0.6]])
    ls = CrossEntropy()
    print("eval:")
    print(ls.eval(y,yhat))
    print("gradient:")
    print(ls.gradient(y,yhat), end = "\n\n")


