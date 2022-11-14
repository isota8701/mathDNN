
import numpy as np
import matplotlib.pyplot as plt

def logistic_fn(x,y, theta):
    return np.log(1+np.exp(-y*np.dot(x,theta)))

def grad_fn(x,y, theta):
    exp = np.exp(-y*np.dot(x,theta))
    return -y * exp / (1 + exp) * x

def SGD(alpha,p, batch_size, N, X, Y, epochs):
    loss = []
    theta = np.random.randn(p)
    for _ in range(epochs):
        indicies = np.random.randint(N, size = batch_size)
        theta-=alpha*np.mean([grad_fn(x,y,theta) for x,y in zip(X[indicies],Y[indicies])], axis = 0)
        loss.append(np.mean([logistic_fn(x,y,theta) for x,y in zip(X, Y)]))
    return theta, loss

N,p = 30,20
np.random.seed(0)
X = np.random.randn(N,p)
Y = 2*np.random.randint(2, size=N)-1
epochs = 3000
theta, loss = SGD(0.01, p, 15, N, X,Y, epochs)
plt.plot(range(epochs), loss)
plt.show()