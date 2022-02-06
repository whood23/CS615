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

    @abstractmethod
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

    def backward(self,gradIn):
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
    
    def backward(self,gradIn):
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
        totRows = np.size(dataIn, axis=0) # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1) # Grab the total number of columns
        tensor = np.random.rand(0,totColumns,totColumns) # Create tensor
        # Preform Gradient calculation
        dataIn[dataIn>=0]=1
        dataIn[dataIn<0]=0
        # Create final matrix and fill tensor
        for row in range(totRows):
            gradData = np.zeros((totColumns,totColumns))
            np.fill_diagonal(gradData,dataIn[row])
            tensor = np.concatenate((tensor,gradData[None]),axis=0)

        return tensor

    def backward(self,gradIn):
        pass
        

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
        totRows = np.size(dataIn, axis=0) # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1) # Grab the total number of columns
        tensor = np.random.rand(0,totColumns,totColumns) # Create an empty tensor
        # Create Preform calculation, final matrix, and tensor append 
        for row in range(totRows):
            diagonal = dataIn[row]*(1-dataIn[row]) # diagonal calculation
            gradData = np.zeros((totColumns,totColumns))
            np.fill_diagonal(gradData,diagonal)
            tensor = np.concatenate((tensor,gradData[None]),axis=0)

        return tensor

    def backward(self,gradIn):
        pass


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        soft_data = np.exp(dataIn)/np.sum(np.exp(dataIn))
        self.setPrevOut(soft_data)
        return soft_data

    def gradient(self):
        dataIn = self.getPrevOut()
        totRows = np.size(dataIn, axis=0) # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1) # Grab the total number of columns
        tensor = np.random.rand(0,totColumns,totColumns) # Create an empty tensor
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

    def backward(self,gradIn):
        pass

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

    def backward(self,gradIn):
        pass


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

    def updateWeights (self,gradIn,eta=0.0001):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        dJdw = (np.matmul(self.getPrevIn().T,gradIn) / gradIn.shape[0])

        self.weights-=eta*dJdw
        self.bias-=eta*dJdb
        pass

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        newset = np.matmul(dataIn,self.weights)+self.bias
        self.setPrevOut(newset)
        return newset
        pass

    def gradient(self):
        weights = self.getWeights()
        grad_fc_data = np.transpose(weights)
        return grad_fc_data

    def backward(self,gradIn,eta=0.0001):
        pass


class LeastSquares:
    def eval(self,y,yhat):
        j = (y - yhat)**2
        return j

    def gradient(self,y,yhat):
        grad_j = -2*(y-yhat)
        return grad_j

    def backward(self,gradIn):
        pass

class LogLoss:
    def eval(self,y,yhat):
        j = (yhat**y)*(1-yhat)**(1-y)
        return j

    def gradient(self,y,yhat):
        grad_j = -(y-yhat)/(yhat*(1-yhat))
        return grad_j

    def backward(self,gradIn):
        pass

class CrossEntropy:
    def eval(self,y,yhat):
        j = np.dot(-y,np.log(np.transpose(yhat)))
        return j

    def gradient(self,y,yhat):
        grad_j = -y/yhat
        return grad_j

    def backward(self,gradIn):
        pass


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


