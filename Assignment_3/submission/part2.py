from matplotlib import pyplot as plt


class VisualGradDescent:
    def __init__(self, eta, epochs, w1, w2, x1, x2):
        self.eta = eta
        self.epochs = epochs
        self.w1 = w1
        self.w2 = w2
        self.x1 = x1
        self.x2 = x2

    def updateWeights(self, weight, grad):
        newWeight = weight - (self.eta * grad)
        return newWeight

    def updateWeightOneGrad(self, weight1, weight2):
        w1Grad = 2 * self.x1 * (self.x1 * weight1 - 5 * self.x2 * weight2 - 2)
        return w1Grad

    def updateWeightTwoGrad(self, weight1, weight2):
        w2Grad = -10 * self.x2 * (self.x1 * weight1 - 5 * self.x2 * weight2 - 2)
        return w2Grad

    def updateJCalc(self, weight1, weight2):
        j = (self.x1 * weight1 - 5 * self.x2 * weight2 - 2) ** 2
        return j

    def runVGD(self):
        weight1 = self.w1
        weight2 = self.w2
        w1 = []
        w2 = []
        J = []

        for epoch in range(self.epochs):
            j = self.updateJCalc(weight1, weight2)
            w1.append(weight1)
            w2.append(weight2)
            J.append(j)
            w1Grad = self.updateWeightOneGrad(weight1, weight2)
            w2Grad = self.updateWeightTwoGrad(weight1, weight2)
            weight1 = self.updateWeights(weight1, w1Grad)
            weight2 = self.updateWeights(weight2, w2Grad)

        return w1, w2, J


if __name__ == '__main__':
    gd = VisualGradDescent(0.01, 100, 0, 0, 1, 1)
    weight1Storage, weight2Storage, jStorage = gd.runVGD()
    ax = plt.axes(projection='3d')
    ax.plot3D(weight1Storage, weight2Storage, jStorage, 'gray')
    ax.set_title('Part 2 Gradient Descent')
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('J')
    plt.show()
