
import numpy as np
import matplotlib.pyplot as plt

def svm_fn(x,y, theta, lam):
    return np.maximum(0, 1 - y * np.dot(x, theta))+ lam * np.dot(theta, theta)

def grad_fn(x,y, theta, lam):
    if 1 - y * np.dot(x, theta) > 0:
        return -y * x + 2 * lam * theta
    else:
        return 2 * lam * theta

def SGD(alpha,p, batch_size, N, X, Y, epochs, lam):
    loss = []
    theta = np.random.randn(p)
    for _ in range(epochs):
        indicies = np.random.randint(N, size = batch_size)
        theta-=alpha*np.mean([grad_fn(x,y,theta, lam) for x,y in zip(X[indicies],Y[indicies])], axis = 0)
        loss.append(np.mean([svm_fn(x,y,theta, lam) for x,y in zip(X, Y)]))
    return theta, loss

N,p = 30,20
np.random.seed(0)
X = np.random.randn(N,p)
Y = 2*np.random.randint(2, size=N)-1
epochs = 3000
lam = 0.1
theta, loss = SGD(0.01, p, 15, N, X,Y, epochs, lam)
plt.plot(range(epochs), loss)
plt.show()