from matplotlib import pyplot as plt
import math


class GradientDescentAdam:
    def __init__(self, x1, w1, graphTitle, fignum, epochs = 100, ro1 = 0.9, ro2 = 0.999, globeLr = 5, smallCst = 1e-8):
        self.x1 = x1
        self.w1 = w1
        self.graphTitle = graphTitle
        self.fignum = fignum
        self.epochs = epochs
        self.ro1 = ro1
        self.ro2 = ro2
        self.globeLr = globeLr
        self.smallCst = smallCst

    def computeGrad(self, W):
        grad = (self.x1 * pow((self.x1 * W), 3)) - ((4 * self.x1) * pow((self.x1 * W), 2)) + ((3 * self.x1) * (self.x1 * W))
        return grad

    def update1stMoment(self, s, grad):
        s = self.ro1 * s + (1 - self.ro1) * grad
        return s

    def update2ndMoment(self, r, grad):
        r = self.ro2 * r + (1 - self.ro2) * (grad * grad)
        return r

    def updateWeight(self, w1, s, r, t):
        w1 = w1 - self.globeLr * ((s / ((1 - pow(self.ro1, t)))) / (math.sqrt(r / (1 - pow(self.ro2, t))) + self.smallCst))
        return w1

    def updateObjective(self, W):
        j = ((1 / 4) * pow((self.x1 * W), 4)) - ((4 / 3) * pow((self.x1 * W), 3)) + ((3 / 2) * pow((self.x1 * W), 2))
        return j

    def runGDAdam(self):
        s = 0
        r = 0
        w = self.w1
        J = []
        E = []
        for epoch in range(self.epochs):
            j = self.updateObjective(w)
            J.append(j)
            E.append(epoch)
            grad = self.computeGrad(w)
            s = self.update1stMoment(s, grad)
            r = self.update2ndMoment(r, grad)
            w = self.updateWeight(w, s, r, epoch+1)
           
        plt.figure(self.fignum)
        plt.plot(E, J)
        plt.legend(["Final Weight:"+ " " + str(w) + "\n"+ "Final Objective:"+ " "+ str(j)])
        plt.ylabel("Objective")
        plt.xlabel("Epoch")
        plt.title(self.graphTitle)
        plt.draw()
        

if __name__ == '__main__':
    plt.style.use("dark_background")
    gda = GradientDescentAdam(1, 0.2, "Part 5: Adaptive Learning Rate", 0)
    gda.runGDAdam()
    plt.show()