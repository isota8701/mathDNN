import numpy as np

N=30
np.random.seed (0)
X = np.random.randn(2,N)
y = np.sign(X [0,:]**2+ X [1 ,:]**2 -0.7)
theta = 0.5
c, s = np.cos( theta ), np.sin( theta )
X = np.array([[c, -s], [s, c]])@X
X = X + np.array([[1] ,[1]])