# Module 1 for forward propagation in Deep Learning
# Module will contain the bones for reusable machine learning code

## Important packages
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

### Edit the forward section
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
        pass

### Edit everything except the Gradient
class InputLayer(Layer):
    def __init__(self,dataIn):
        ##empty
        pass

    def forward(self,dataIn):
        ##empty
        pass

    def gradient(self):
        pass

# Activation Layers
### Edit all the Layer classes

class Linear(Layer):
    def __init__(self):
        ##empty
        pass
    
    def forward(self,dataIn):
        ##empty
        pass

    def gradient(self):
        pass

class ReLu(Layer):
    def __init__(self):
        ##empty
        pass
    
    def forward(self,dataIn):
        ##empty
        pass

    def gradient(self):
        pass

class Sigmoid(Layer):
    def __init__(self):
        ##empty
        pass
    
    def forward(self,dataIn):
        ##empty
        pass

    def gradient(self):
        pass

class Softmax(Layer):
    def __init__(self):
        ##empty
        pass
    
    def forward(self,dataIn):
        ##empty
        pass

    def gradient(self):
        pass

class Tan(Layer):
    def __init__(self):
        ##empty
        pass
    
    def forward(self,dataIn):
        ##empty
        pass

    def gradient(self):
        pass


# Fully Connected Layer
### Edit all defs except the gradient

class FullyConnectedLAyer(Layer):
    def __init__(self, sizeIn, sizeOut):
        ##empty
        pass

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
