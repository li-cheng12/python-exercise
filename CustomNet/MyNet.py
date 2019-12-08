import numpy as np
import pickle

np.random.seed(12345)
# output, input
net_structure = [[10, 20],[30, 10],[10, 30],[2, 10]]
weights = []
dWeights = {}
bias = []
dBias = {}
z = []
a = []
test_set = np.random.randint(1000, size = (1, 100))
def init(n_feature, label_size):
    net_structure = [[2, n_feature],[3, 2],[1, 3],[label_size, 1]]
    for i in range(len(net_structure)):
        weights.append(np.random.rand(net_structure[i][0], net_structure[i][1]))
        bias.append(np.random.rand(net_structure[i][0], 1))

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def dSigmoid(da, Z):
     return da * sigmoid(Z)*(1- sigmoid(Z))

def relu(Z):
    return np.maximum(0, Z)

def dRelu(da, Z):
    dz = np.array(da, copy=True)
    dz[Z < 0] = 0
    return dz

def single_forward(X, layerIdx):
    W = weights[layerIdx]
    B = bias[layerIdx]
    Z = np.dot(W, X) + B
    return Z

def forward(X):
    # X: 20, 30
    X_tmp = np.array(X, copy=True)
    for i in range(len(net_structure)):
        Z = single_forward(X_tmp, layerIdx=i)
        A = relu(Z)
        z.append(Z)
        a.append(A)
        X_tmp = A
    return X_tmp

def forward_interface(X):
    X_tmp = np.array(X, copy=True)
    for i in range(len(net_structure)):
        Z = single_forward(X_tmp, layerIdx=i)
        A = relu(Z)
        X_tmp = A
    return A

def single_backward(da, layerIdx):
    dz = dRelu(da, z[layerIdx])
    # w : outputsize, inputsize
    # dw: outputsize, inputsize
    # preA:inputsize, batch
    # dz: outputsize, batch
    # A : outputsize, batch
    # B: outputsize, 1
    m = a[layerIdx - 1].shape[1]
    dWeights[layerIdx] = np.dot(dz, a[layerIdx - 1].T)/m
    dBias[layerIdx] = np.dot(dz, np.ones((dz.shape[1], 1)))/m
    dpreA = np.dot(weights[layerIdx].T, dz)
    return dpreA

def backward(y, yPre):
    # y : outputsize, batch
    # dyPre = -(np.divide(y, yPre) - np.divide(1 - y, 1 - yPre))
    dyPre = yPre - y
    for i in reversed(range(len(net_structure))):
        dyPre = single_backward(dyPre, i)

def get_accuracy_value(Y_hat, Y):
    return (Y_hat == Y).all(axis=0).mean()

def loss_function(Y_hat, Y):
    loss_value = (Y - Y_hat) * (Y - Y_hat) / 2
    return np.mean(loss_value)

def update(learning_rate):
    for i in range(len(net_structure)):
        weights[i] -= learning_rate * dWeights[i]
        bias[i] -= learning_rate * dBias[i]

def train(X, Y, learn_rate):
    init(X.shape[0], Y.shape[0])
    for i in range(5000):
        Y_hat = forward(X)
        backward(Y, Y_hat)
        update(learn_rate)
        if i % 500 == 0:
            test()


def test():
    test_y = 3 * test_set
    pre = forward_interface(test_set)
    loss = loss_function(pre, test_y)
    print(" loss:", loss)

if __name__ == '__main__':
    X = np.random.randint(1000, size = (1, 100)) / 10000000
    noise = np.random.random_sample(size = (1,100)) / 1000000000
    Y = 3 * X + noise
    train(X, Y, 0.00000001)

    test_y = 3 * test_set
    pre = forward_interface(test_set)
    # for i in range(len(test_set[0])):
    #     print(str(test_set[0][i]), ":" + str(pre[0][i]))




























