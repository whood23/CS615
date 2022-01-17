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
        relu_data = np.maximum(0,dataIn)
        self.setPrevOut(relu_data)
        return relu_data

    def gradient(self):
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
        pass

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        top, bottom = np.exp(dataIn-(np.amax(dataIn, axis=0))), np.sum(np.exp(dataIn-(np.amax(dataIn, axis=0))))
        soft_data = top/bottom
        self.setPrevOut(soft_data)
        return soft_data

    def gradient(self):
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

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        newset = np.matmul(dataIn,self.weights)+self.bias
        self.setPrevOut(newset)
        return newset

    def gradient(self):
        pass


if __name__ == '__main__':
    test_set = np.array([[1,2,3,4],[5,6,7,8]])
    print()
    print("*************************************************************************")
    print("****************** Part 5: Testing the layers ***************************")
    print("*************************************************************************")
    il = InputLayer(test_set)
    fl = FullyConnectedLayer(4,2)
    relu = ReLuLayer()
    sig = SigmoidLayer()
    sm = SoftmaxLayer()
    tanh = TanhLayer()
    print("Input Layer:")
    print(il.forward(test_set),end="\n\n")
    print("Fully Connected Layer:")
    print(fl.forward(test_set),end="\n\n")
    print("ReLu Activation Layer:")
    print(relu.forward(test_set),end="\n\n")
    print("Sigmoid Activation Layer:")
    print(sig.forward(test_set),end="\n\n")
    print("SoftMax Activation Layer:")
    print(sm.forward(test_set),end="\n\n")
    print("Tanh Activation Layer:")
    print(tanh.forward(test_set),end="\n\n")

    print("*************************************************************************")
    print("*********** Part 6: Connecting Layers and Foward Propagate **************")
    print("*************************************************************************")
    test_set = np.array([[1,2,3,4],[5,6,7,8]])
    il = InputLayer(test_set)
    il_data = il.forward(test_set)
    fl = FullyConnectedLayer(4,2)
    fl_data = fl.forward(il_data)
    sig = SigmoidLayer()
    sig_data = sig.forward(fl_data)
    print("Input Layer:")
    print(il_data,end="\n\n")
    print("Fully Connected Layer:")
    print(fl_data,end="\n\n")
    print("Sigmoid Activation Layer:")
    print(sig_data,end="\n\n")

    print("*************************************************************************")
    print("***************** Part 7: Testing on full dataset ***********************")
    print("*************************************************************************")
    il = InputLayer(np.genfromtxt('mcpd_augmented.csv', delimiter=','))
    il_data = il.forward(np.genfromtxt('mcpd_augmented.csv', delimiter=','))
    fl = FullyConnectedLayer(6,2)
    fl_data = fl.forward(il_data)
    sig = SigmoidLayer()
    sig_data = sig.forward(fl_data)
    print("Sigmoid Activation Layer:")
    print(sig_data,end="\n\n")