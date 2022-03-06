class FullyConnected(Layer):

    def __init__(self, sizein, sizeout):
        super().__init__()
        self._FullyConnected__weights = 0.0001 * (np.random.rand(sizein, sizeout) - 0.5)
        self._FullyConnected__biases = 0.0001 * (np.random.rand(1, sizeout) - 0.5)
        self._FullyConnected__sW = 0
        self._FullyConnected__rW = 0
        self._FullyConnected__sb = 0
        self._FullyConnected__rb = 0
        self._FullyConnected__useBias = useBias

        

    def setWeights(self, weights):
        self._FullyConnected__weights = weights

    def getWeights(self):
        return self._FullyConnected__weights

    def setBias(self, bias):
        self._FullyConnected__biases = bias

    def getBias(self):
        return self._FullyConnected__bias

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        if self._FullyConnected__useBias == 1:
            temp = dataIn @ self._FullyConnected__weights + self._FullyConnected__biases
        else:
            temp = dataIn @ self._FullyConnected__weights
        self.setPrevOut(temp)
        return temp

    def backward(self, gradIn):
        return gradIn @ self._FullyConnected__weights.T

    def updateWeights(self, gradIn, eta, p1=0.9, p2=0.999, rho=-14, epoch=-1):
        reg = 0
        pi = self.getPrevIn()
        po = self.getPrevOut()
        dJdW = pi.T @ gradIn
        deltaJ = dJdW / gradIn.shape[0] + 2 * reg * self._FullyConnected__weights / gradIn.shape[0]
        self._FullyConnected__sW = p1 * self._FullyConnected__sW + (1 - p1) * deltaJ
        self._FullyConnected__rW = p2 * self._FullyConnected__rW + (1 - p2) * (deltaJ * deltaJ)
        dJdb = np.sum(gradIn, 0)
        deltaJ = dJdb / gradIn.shape[0] + 2 * reg * self._FullyConnected__biases / gradIn.shape[0]
        self._FullyConnected__sb = p1 * self._FullyConnected__sb + (1 - p1) * deltaJ
        self._FullyConnected__rb = p2 * self._FullyConnected__rb + (1 - p2) * (deltaJ * deltaJ)
        if epoch == -1:
            self._FullyConnected__weights -= eta * dJdW / pi.shape[0]
            self._FullyConnected__biases -= eta * dJdb / pi.shape[0]
        else:
            self._FullyConnected__weights -= eta * (self._FullyConnected__sW / (1 - p1 ** epoch) / (np.sqrt(self._FullyConnected__rW / (1 - p2 ** epoch)) + rho) + 2 * reg * self._FullyConnected__weights / gradIn.shape[0])
            self._FullyConnected__biases -= eta * (self._FullyConnected__sb / (1 - p1 ** epoch) / (np.sqrt(self._FullyConnected__rb / (1 - p2 ** epoch)) + rho) + 2 * reg * self._FullyConnected__biases / gradIn.shape[0])

    def gradient(self):
        return np.tile(self._FullyConnected__weights.T, (self.getPrevIn().shape[0], 1, 1))
