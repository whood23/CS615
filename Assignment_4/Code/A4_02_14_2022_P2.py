from matplotlib import pyplot as plt
import numpy


class VisualObjectiveFunction:
    def __init__(self, x1, w1Begin, w1End, w1Inc = 0.1):
        self.w1Begin = w1Begin
        self.w1End = w1End
        self.w1Inc = w1Inc
        self.x1 = x1

    def computeObjective(self, weight):
        j = (1 / 4) * pow((self.x1 * weight), 4) - (4 / 3) * pow((self.x1 * weight), 3) + (3 / 2) * pow(
            (self.x1 * weight), 2)
        return j

    def runVOJ(self):
        W = []
        J = []
        for i in numpy.arange(self.w1Begin, self.w1End, self.w1Inc):
            j = self.computeObjective(i)
            W.append(i)
            J.append(j)

        plt.style.use("dark_background")
        plt.plot(W, J)
        plt.ylabel("Weight")
        plt.xlabel("J")
        plt.title("Part 2: Visualizing an Objective Function")
        plt.draw()


if __name__ == '__main__':
    vof = VisualObjectiveFunction(1, -2, 5, 0.1)
    vof.runVOJ()
    plt.show()