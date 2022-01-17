### CS615 - Deep Learning - Assignment 1

I tested my code using the provided scrips below. 

I saved the fill code in the .py file. If you want to run the script on tux all you need to do is just run the script using python3.

ex. python3 hw1.py

*Note the mcpd_augmented.csv file needs to be in the same folder with the .py script in order to run.


Script used to execute the code in the hw1.py file.

```
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
    print("Input Layer")
    print(il.forward(test_set),end="\n\n")
    print("Fully Connected Layer")
    print(fl.forward(test_set),end="\n\n")
    print("ReLu Activation Layer")
    print(relu.forward(test_set),end="\n\n")
    print("Sigmoid Activation Layer")
    print(sig.forward(test_set),end="\n\n")
    print("SoftMax Activation Layer")
    print(sm.forward(test_set),end="\n\n")
    print("Tanh Activation Layer")
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
    print("Input Layer")
    print(il_data,end="\n\n")
    print("Fully Connected Layer")
    print(fl_data,end="\n\n")
    print("Sigmoid Activation Layer")
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
    print("Sigmoid Activation Layer")
    print(sig_data,end="\n\n")
```