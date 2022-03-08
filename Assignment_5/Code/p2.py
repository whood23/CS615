'''
Import Modules
'''
import numpy as np
import hw5_layers as imp
import OneHotEncoder as ohe
import matplotlib.pyplot as plt
import time as dt

Script_start_time = dt.time()
'''
Split Data
'''
# Train Data
data = np.genfromtxt('mnist_train_100.csv', delimiter=',', skip_header=True)
np.random.shuffle(data)
XTrain = data[:, 1:]
YTrain = data[:, :1]
oneHot = ohe.OneHotEncoder(YTrain.astype(int))
YTrainohe = oneHot.reformatData()

# Test Data
dataTest = np.genfromtxt('mnist_valid_10.csv', delimiter=',', skip_header=True)
np.random.shuffle(dataTest)
XTest = dataTest[:, 1:]
YTest = dataTest[:, :1]
oneHot = ohe.OneHotEncoder(YTest.astype(int))
YTestohe = oneHot.reformatData()

'''
Archetecture
'''
L1 = imp.InputLayer(XTrain)
L2 = imp.FullyConnected(XTrain.shape[1], 10)
L3 = imp.SigmoidLayer()
L4 = imp.LogLoss()
layers = [L1, L2, L3, L4]

'''
Training and testing
'''
# Variables
epochs = 100
Y_train = YTrainohe
Y_test = YTestohe
objStorage = 0
endDiff = 1e-10
objEvalTrain = []
objEvalTest = []

# Run Epochs
for j in range(epochs):
    start_time = dt.time()
    H_train = XTrain
    # Forward run
    for i in range(len(layers) - 1):
        H_train = layers[i].forward(H_train)

    # Backwards run
    grad = layers[-1].gradient(Y_train, H_train)
    for i in range(len(layers) - 2, 0, -1):
        newGrad = layers[i].backward(grad)
        if isinstance(layers[i], imp.FullyConnected):
            layers[i].updateWeights2(grad, j)
        grad = newGrad

    # Evaluation
    objEval_train = np.mean(layers[-1].eval(Y_train, H_train))
    objEvalTrain.append(objEval_train)

    ## Validation
    H_test = XTest
    # Forward run
    for i in range(len(layers) - 1):
        H_test = layers[i].forward(H_test)

    # Evaluation
    objEval_test = np.mean(layers[-1].eval(Y_test, H_test))
    objEvalTest.append(objEval_test)

    # Loop Break
    diff = np.absolute(objEval_train - objStorage)
    if diff < endDiff:
        break
    objStorage = objEval_train

    end_time = dt.time()
    print("Epoch {} Duration: {}".format((j+1), (end_time - start_time)))


H = XTrain
for i in range(len(layers) - 1):
    H = layers[i].forward(H)


classification = H
classify =  np.argmax(classification, axis=1).reshape(classification.shape[0], 1)
Classify = (YTrain == classify)
print("Training Accuracy: {0:.2f}%".format((np.count_nonzero(Classify) / np.size(YTrain, axis=0)) * 100))

H = XTest
for i in range(len(layers) - 1):
    H = layers[i].forward(H)


classificationTest = H
classifyTest =  np.argmax(classificationTest, axis=1).reshape(classificationTest.shape[0], 1)
ClassifyTest = (YTest == classifyTest)
print("Testing Accuracy: {0:.2f}%".format((np.count_nonzero(ClassifyTest) / np.size(YTest, axis=0)) * 100))

Script_end_time = dt.time()
print("Total Duration: {}".format(Script_end_time - Script_start_time))

'''
Plot
'''

# Display results
epochs = list(range(0, 100))
plt.figure(1)
plt.plot(epochs, objEvalTrain)
plt.plot(epochs, objEvalTest)
plt.legend([("Final training accuracy:" + " " + (str(round(((np.count_nonzero(Classify) / np.size(YTrain, axis=0)) * 100), 2))) + "%"), ("Final test accuracy:" + " " + (str(round(((np.count_nonzero(ClassifyTest) / np.size(YTest, axis=0)) * 100), 2))) + "%")])
plt.ylabel("Objective")
plt.xlabel("Epoch")
plt.title("Part 2 Fitting")
plt.draw()

plt.show()