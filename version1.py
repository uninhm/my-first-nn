import numpy as np
import random as rnd

data = np.load("mnist.npz")

X = 1/255 * data["X"]
Y = data["y"]

sigmoid = lambda x: 1/(1 + np.exp(-x))
sigmoid_prime = lambda x: x * (1 - x)

class NN:
    layer_sizes = [28*28, 16, 16, 10]
    n_layers = len(layer_sizes)

    def __init__(self):
        self.layers = [ np.ndarray((sz,), dtype=np.float64) for sz in self.layer_sizes ]

        self.weights = []
        for i in range(self.n_layers-1):
            self.weights.append(
                    np.random.random((self.layer_sizes[i+1], self.layer_sizes[i]))-0.5
                    )

        self.biases = []
        for i in range(self.n_layers-1):
            self.biases.append(
                    np.random.random((self.layer_sizes[i+1],))-0.5
                    )

    def eval(self, x: np.ndarray) -> np.ndarray:
        self.layers[0][:] = x
        for i in range(self.n_layers-1):
            self.layers[i+1][:] = sigmoid(self.weights[i] @ self.layers[i] + self.biases[i])
        return self.layers[-1]


    def cost(self, x: np.ndarray, y: np.ndarray):
        return np.sum((self.eval(x) - y)**2)

    def back_propagation(self, x: np.ndarray, y: np.ndarray):
        ev = self.eval(x)
        # dC/da
        dlayers = [ None ] * self.n_layers
        dlayers[-1] = 2 * (ev - y)
        # dC/dW
        dweights = [ None ] * (self.n_layers-1)
        # dC/db
        dbiases = [ None ] * (self.n_layers-1)

        for L in range(self.n_layers-2, -1, -1):
            A = dlayers[L+1] * sigmoid_prime(self.layers[L+1])
            dweights[L] = A.reshape((-1, 1)) @ self.layers[L].reshape((1, -1))
            dbiases[L] = A
            dlayers[L] = np.transpose(self.weights[L]) @ A

        return dweights, dbiases, np.sum((ev-y)**2)

    def gradient_descent(self, X, Y, learning_rate=0.01, sample_size=20, epochs=10):
        data = list(zip(list(X), list(Y)))
        for epoch in range(epochs):
            rnd.shuffle(data)
            for j in range(0, len(data), sample_size):
                avgW = [ np.zeros(W.shape, dtype=np.float64) for W in self.weights ]
                avgb = [ np.zeros(b.shape, dtype=np.float64) for b in self.biases ]
                avgC = 0
                for i in range(j, j+sample_size):
                    x, y = X[i], Y[i]
                    y_vec = np.zeros(10)
                    y_vec[int(y)] = 1.
                    dw, db, C = self.back_propagation(x, y_vec)
                    avgC += 1/sample_size * C
                    for L in range(self.n_layers-1):
                        avgW[L] += 1/sample_size * dw[L]
                        avgb[L] += 1/sample_size * db[L]
                for L in range(self.n_layers-1):
                    self.weights[L] -= learning_rate * avgW[L]
                    self.biases[L] -= learning_rate * avgb[L]
                print(f'epoch {epoch}/{epochs}: {avgC}')

    def save(self, filename="nn_params.npz"):
        np.savez(
            filename,
            weights=np.array(self.weights, dtype=object),
            biases =np.array(self.biases,  dtype=object)
        )

    def load(self, filename="nn_params.npz"):
        data = np.load(filename, allow_pickle=True)
        # convert object-array back to list-of-arrays
        self.weights = [arr for arr in data["weights"]]
        self.biases  = [arr for arr in data["biases"]]


if __name__ == "__main__":
    nn = NN()
    # nn.load('version1_params.npz')
    nn.gradient_descent(X, Y, learning_rate=0.005, sample_size=10, epochs=100)
    print(nn.eval(X[0]), Y[0])
    # nn.save('version1_params.npz')
