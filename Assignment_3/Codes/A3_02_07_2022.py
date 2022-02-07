## Important packages
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

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
        sg = self.gradient()
        grad = np.zeros((gradIn.shape[0], sg.shape[2]))
        for n in range(gradIn.shape[0]):
            grad[n, :] = np.matmul(gradIn[n, :], sg[n, :, :])

        return grad

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
        dataIn = np.copy(self.getPrevIn())
        totRows = np.size(dataIn, axis=0) # Grab the total number of rows
        totColumns = np.size(dataIn, axis=1) # Grab the total number of columns
        tensor = np.random.rand(0,totColumns,totColumns) # Create tensor
        # Preform Gradient calculation
        dataIn[dataIn>=0]=1
        dataIn[dataIn<0]=1
        # Create final matrix and fill tensor
        for row in range(totRows):
            gradData = np.zeros((totColumns,totColumns))
            np.fill_diagonal(gradData,dataIn[row])
            tensor = np.concatenate((tensor,gradData[None]),axis=0)

        return tensor



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



# Fully Connected Layer
### Edit all defs except the gradient

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut):
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
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

    def updateWeights(self,gradIn,eta=0.0001):
        dJdb = np.sum(gradIn, axis=0) / gradIn.shape[0]
        dJdw = (np.matmul(self.getPrevIn().T,gradIn) / gradIn.shape[0])
        self.bias = self.bias - (eta * dJdb)
        self.weights = self.weights - (eta * dJdw)

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        newset = np.matmul(dataIn,self.weights)+self.bias
        self.setPrevOut(newset)
        return newset

    def gradient(self):
        weights = self.getWeights()
        grad_fc_data = np.transpose(weights)
        return grad_fc_data

    def backward(self,gradIn):
        sg = self.gradient()
        grad = np.matmul(gradIn, sg)
        return grad

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
        j = -1 * ((np.log(yhat+10e-7)*y)+(1-y)*np.log(1-yhat + 10e-7))
        return j

    def gradient(self,y,yhat):
        grad_j = -(y-yhat)/(yhat*(1-yhat) + 10e-7)
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
    def split(data, percent):
        totRows = np.size(data, axis=0)
        X, Y = data[:round(totRows * percent), :], data[round(totRows * percent):, :]
        return X, Y

    percent = 2 / 3
    np.random.seed(0)
    """
    Part 4
    """
    data = np.genfromtxt('mcpd_augmented.csv', delimiter=',', skip_header= True)
    np.random.shuffle(data)  # shuffle data
    first_half,second_half = split(data, percent)
    train_data = first_half[:, :-1]
    train_comp = first_half[:, -1:]
    test_data = second_half[:, :-1]
    test_comp = second_half[:, -1:]

    L1 = InputLayer(train_data)
    L2 = FullyConnectedLayer(train_data.shape[1],1)
    L3 = LeastSquares()

    layers = [L1, L2, L3]

    epochs = 10000
    storage = []
    epoch_storage = []
    eta = 0.0001
    prev_check = 0
    endDiff = 1 * (10 ** -10)
    for j in range(epochs):
        H = train_data
        # Forward
        for i in range(len(layers) - 1):
            H = layers[i].forward(H)

        # run gradient on the eval layer
        grad = layers[-1].gradient(train_comp, H)
        # Backwards
        for i in range(len(layers) - 2, 0, -1):
            newgrad = layers[i].backward(grad)

            if isinstance(layers[i], FullyConnectedLayer):
                layers[i].updateWeights(grad, eta)

            grad = newgrad
        # Storing each evaluation in a list
        check = np.mean(layers[-1].eval(train_comp, H))

        storage.append(check)
        epoch_storage.append(j)
        if np.absolute(check - prev_check) < endDiff:
            break

        prev_check = check


    """
    Part 5
    """
    data = np.genfromtxt('KidCreative.csv', delimiter=',', skip_header= True)
    np.random.shuffle(data)  # shuffle data
    first_half,second_half = split(data, percent)

    train_data = first_half[:, 2:]
    train_comp = first_half[:, 1:2]
    test_data = second_half[:, 2:]
    test_comp = second_half[:, 1:2]

    L1 = InputLayer(train_data)
    L2 = FullyConnectedLayer(train_data.shape[1],1)
    L3 = SigmoidLayer()
    L4 = LogLoss()

    layers = [L1,L2,L3,L4]

    epochs = 10000
    storage = []
    epoch_storage = []
    eta = 0.0001
    prev_check = 0
    endDiff = 1 * (10 ** -10)
    for j in range(epochs):
        H = train_data
        # Forward
        for i in range(len(layers)-1):
            H = layers[i].forward(H)

        # run gradient on the eval layer
        grad = layers[-1].gradient(train_comp, H)
        # Backwards
        for i in range(len(layers)-2,0,-1):
            newgrad = layers[i].backward(grad)

            if isinstance(layers[i], FullyConnectedLayer):
                layers[i].updateWeights(grad,eta)

            grad = newgrad
        # Storing each evaluation in a list
        check = np.mean(layers[-1].eval(train_comp, H))

        storage.append(check)
        epoch_storage.append(j)
        if np.absolute(check - prev_check) < endDiff:
            break

        prev_check = check

    plt.plot(epoch_storage, storage)
    plt.title('Epoch vs Log Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')


    L1 = InputLayer(test_data)
    L2 = FullyConnectedLayer(test_data.shape[1],1)
    L3 = SigmoidLayer()
    L4 = LogLoss()

    layers = [L1,L2,L3,L4]

    epochs = 10000
    storage2 = []
    epoch_storage2 = []
    eta = 0.0001
    prev_check = 0
    endDiff = 1 * (10 ** -10)
    for j in range(epochs):
        H = test_data
        # Forward
        for i in range(len(layers)-1):
            H = layers[i].forward(H)

        # run gradient on the eval layer
        grad = layers[-1].gradient(test_comp, H)
        # Backwards
        for i in range(len(layers)-2,0,-1):
            newgrad = layers[i].backward(grad)

            if isinstance(layers[i], FullyConnectedLayer):
                layers[i].updateWeights(grad,eta)

            grad = newgrad
        # Storing each evaluation in a list
        check = np.mean(layers[-1].eval(test_comp, H))

        storage2.append(check)
        epoch_storage2.append(j)
        if np.absolute(check - prev_check) < endDiff:
            break

        prev_check = check

    plt.plot(epoch_storage2, storage2)
    plt.legend(['Training','Validation'])
    plt.show()


