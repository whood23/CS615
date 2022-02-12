from matplotlib import pyplot as plt


class ModelInitialEffects:
    def __init__(self, w, x1, graphTitle, fignum, epochs=100, eta=0.1):
        self.w = w
        self.x1 = x1
        self.epochs = epochs
        self.eta = eta
        self.graphTitle = graphTitle
        self.fignum = fignum

    def updateWeight(self, W, grad):
        newW = W - (self.eta * grad)
        return newW

    def calculateGrad(self, W):
        grad = (self.x1 * pow((self.x1 * W), 3)) - ((4 * self.x1) * pow((self.x1 * W), 2)) + ((3 * self.x1) * (self.x1 * W))
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

        plt.figure(self.fignum)
        plt.plot(E, J)
        plt.legend(["Final Weight:"+ " " + str(w1) + "\n"+ "Final Objective:"+ " "+ str(j)])
        plt.ylabel("Objective")
        plt.xlabel("Epoch")
        plt.title(self.graphTitle)
        plt.draw()

        return w1, j


if __name__ == '__main__':
    print("Final Results")
    print()

    simulations = ["Part 4: @eta = 0.001", "Part 4: @eta = 0.01", "Part 4: @eta = 1.0", "Part 4: @eta = 5.0"]
    parametersEta = [0.001, 0.01, 1.0, 5.0]

    plt.style.use("dark_background")

    for i in range(len(simulations)):
        if i < 3:
            mie = ModelInitialEffects(0.2, 1, simulations[i], i, 100, parametersEta[i])
            weight, j = mie.runMIE()

        elif i == 3:
            mie = ModelInitialEffects(0.2, 1, simulations[i], i, 6, parametersEta[i])
            weight, j = mie.runMIE()

        print(simulations[i][7:])
        print("Final Weight:", weight)
        print("Final Objective:", j)
        print()
        
    plt.show()