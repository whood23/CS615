import numpy as np


class OneHotEncoder:
    def __init__(self, targetData):
        self.targetData = targetData

    def reformatData(self):
        data = self.targetData
        encodedData = np.eye(data.max()+1)[data]

        return encodedData

if __name__ == '__main__':
    data = np.genfromtxt('mnist_train_100.csv', delimiter=',')
    ohe = OneHotEncoder(data[:, :1].astype(int))
    print(ohe.reformatData())
