from matplotlib import pyplot as plt


class ModelInitialEffects:
    def __init__(self, w, x1, graphTitle, epochs=100, eta=0.1):
        self.w = w
        self.x1 = x1
        self.epochs = epochs
        self.eta = eta
        self.graphTitle = graphTitle

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
        w1 = self.w
        E = []
        J = []

        for epoch in range(self.epochs):
            j = self.calculateObjective(w1)
            J.append(j)
            E.append(epoch)
            grad = self.calculateGrad(w1)
            w1 = self.updateWeight(w1, grad)

        plt.style.use("dark_background")
        plt.plot(E, J)
        plt.ylabel("Objective")
        plt.xlabel("Epoch")
        plt.title(self.graphTitle)
        plt.show()

        return w1, j


if __name__ == '__main__':
    print("Final Results")
    print()

    mie = ModelInitialEffects(0.2, 1, "Part 3: @eta = 0.001", 100, 0.001)
    weight, obj = mie.runMIE()
    print("@eta=0.001")
    print("Final Weight:", weight)
    print("Final Objective:", obj)
    print()

    mie = ModelInitialEffects(0.2, 1, "Part 3: @eta = 0.01", 100, 0.01)
    weight, obj = mie.runMIE()
    print("@eta=0.01")
    print("Final Weight:", weight)
    print("Final Objective:", obj)
    print()

    mie = ModelInitialEffects(0.2, 1, "Part 3: @eta = 1.0", 100, 1.0)
    weight, obj = mie.runMIE()
    print("@eta=1.0")
    print("Final Weight:", weight)
    print("Final Objective:", obj)
    print()

    mie = ModelInitialEffects(0.2, 1, "Part 3: @eta = 5.0", 100, 5.0)
    weight, obj = mie.runMIE()
    print("@eta=5.0")
    print("Final Weight:", weight)
    print("Final Objective:", obj)
    print()
