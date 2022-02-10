from matplotlib import pyplot as plt


class ModelInitialEffects:
    def __init__(self, w1, x1, graphTitle, fignum, epochs=100, eta=0.1):
        self.w1 = w1
        self.x1 = x1
        self.epochs = epochs
        self.eta = eta
        self.graphTitle = graphTitle
        self.fignum = fignum

    def updateWeight(self, W, grad):
        newW = W - (self.eta * grad)
        return newW

    def calculateGrad(self, W):
        grad = (self.x1 * pow((self.x1 * W), 3)) - ((4 * self.x1) * pow((self.x1 * W), 2)) + (
                    (3 * self.x1) * (self.x1 * W))
        return grad

    def calculateObjective(self, W):
        j = ((1 / 4) * pow((self.x1 * W), 4)) - ((4 / 3) * pow((self.x1 * W), 3)) + ((3 / 2) * pow((self.x1 * W), 2))
        return j

    def runMIE(self):
        weight = self.w1
        E = []
        J = []

        for epoch in range(self.epochs):
            j = self.calculateObjective(weight)
            J.append(j)
            E.append(epoch)
            grad = self.calculateGrad(weight)
            weight = self.updateWeight(weight, grad)

        plt.figure(self.fignum)
        plt.style.use("dark_background")
        plt.plot(E, J)
        plt.ylabel("Objective")
        plt.xlabel("Epoch")
        plt.title(self.graphTitle)
        plt.draw()

        return weight, j


if __name__ == '__main__':
    print("Final Results")
    print()

    mie = ModelInitialEffects(-1, 1, "Part 3: @w = -1", 1)
    weight, j = mie.runMIE()
    print("@w=-1")
    print("Final Weight:", weight)
    print("Final Objective:", j)
    print()

    mie = ModelInitialEffects(0.2, 1, "Part 3: @w = 0.2", 2)
    weight, j = mie.runMIE()
    print("@w=0.2")
    print("Final Weight:", weight)
    print("Final Objective:", j)
    print()

    mie = ModelInitialEffects(0.9, 1, "Part 3: @w = 0.9", 3)
    weight, j = mie.runMIE()
    print("@w=0.9")
    print("Final Weight:", weight)
    print("Final Objective:", j)
    print()

    mie = ModelInitialEffects(4, 1, "Part 3: @w = 4", 4)
    weight, j = mie.runMIE()
    print("@w=4")
    print("Final Weight:", weight)
    print("Final Objective:", j)
    print()

    plt.show()