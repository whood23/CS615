# Module 1 for forward propagation in Deep Learning
# Module will contain the bones for reusable machine learning code

from abc import ABC, abstractmethod
class Layer(ABC):
    def __init__(ABC):
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

class InputLayer(Layer):
    def __init__(self,dataIn):
        ##empty

    def forward(self,dataIn):
        ##empty

    def gradient(self):
        pass

# Activation layers

class Linear(Layer):
    def __init__(self):
        ##empty
    
    def forward(self,dataIn):
        ##empty

    def gradient(self):
        pass

class ReLu(Layer):
    def __init__(self):
        ##empty
    
    def forward(self,dataIn):
        ##empty

    def gradient(self):
        pass

class Sigmoid(Layer):
    def __init__(self):
        ##empty
    
    def forward(self,dataIn):
        ##empty

    def gradient(self):
        pass

class Softmax(Layer):
    def __init__(self):
        ##empty
    
    def forward(self,dataIn):
        ##empty

    def gradient(self):
        pass

class Tan(Layer):
    def __init__(self):
        ##empty
    
    def forward(self,dataIn):
        ##empty

    def gradient(self):
        pass


# Fully Connected Layer

class FullyConnectedLAyer(Layer):
    def __init__(self, sizeIn, sizeOut):
        ##empty

    def getWeights(self):
        ##empty
        pass

    def setWeights(self,weights):
        ##empty
        pass

    def getBias(self):
        ##empty
        pass

    def setBias(self,bias):
        ##empty
        pass

    def forward(self,dataIn):
        ##empty
        pass
    
    def gradient(self):
        pass
