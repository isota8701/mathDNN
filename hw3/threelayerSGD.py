import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

alpha = 0.1
K = 1000
B = 128
N = 512

def f_true(x) :
    return (x-2) * np.cos(x*4)

torch.manual_seed(0)
X_train = torch.normal(0.0, 1.0, (N,))
y_train = f_true(X_train) + torch.normal(0,0.5, X_train.shape)
X_val = torch.normal(0.0, 1.0, (N//5,))
y_val = f_true(X_val)

train_dataloader = DataLoader(TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1)), batch_size=B)
test_dataloader = DataLoader(TensorDataset(X_val.unsqueeze(1), y_val.unsqueeze(1)), batch_size=B)

'''
unsqueeze(1) reshapes the data into dimension [N,1],
where is 1 the dimension of an data point.

The batchsize of the test dataloader should not affect the test result
so setting batch_size=N may simplify your code.
In practice, however, the batchsize for the training dataloader
is usually chosen to be as large as possible while not exceeding
the memory size of the GPU. In such cases, it is not possible to
use a larger batchsize for the test dataloader.
'''

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(1, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,1)
        self.actv = nn.Sigmoid()
    def forward(self, x):
        x = self.actv(self.l1(x))
        x = self.actv(self.l2(x))
        x = self.l3(x)
        return x



model = MLP()
model.l1.weight.data = torch.normal(0,1, model.l1.weight.shape)
model.l1.bias.data = torch.full(model.l1.bias.shape, 0.03)
model.l2.weight.data = torch.normal(0,1, model.l2.weight.shape)
model.l2.bias.data = torch.full(model.l2.bias.shape, 0.03)
model.l3.weight.data = torch.normal(0,1, model.l3.weight.shape)
model.l3.bias.data = torch.full(model.l3.bias.shape, 0.03)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)
train_loss, valid_loss = [], []
for e in range(K):
    tmp_train_loss = []
    for x,y in train_dataloader:
        out = model(x)
        loss = loss_fn(out, y)
        tmp_train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss.append(np.average(tmp_train_loss))
    tmp_valid_loss = []
    for x,y in test_dataloader:
        out = model(x)
        loss = loss_fn(out, y)
        tmp_valid_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    valid_loss.append(np.average(tmp_valid_loss))

    if (e+1)%100 ==0:
        print(f"epoch {e + 1} / {K}")
        print(f"\ttrain loss: {train_loss[-1]:.5f}")
        print(f"\tval loss: {valid_loss[-1]:.5f}")
with torch.no_grad():
    xx = torch.linspace(-2,2,1024).unsqueeze(1)
    plt.plot(X_train,y_train,'rx',label='Data points')
    plt.plot(xx,f_true(xx),'r',label='True Fn')
    plt.plot(xx, model(xx),label='Learned Fn')
plt.legend()
plt.show()

'''
When plotting torch tensors, you want to work with the
torch.no_grad() context manager.

When you call plt.plot(...) the torch tensors are first converted into
numpy arrays and then the plotting proceeds.
However, our trainable model has requires_grad=True to allow automatic
gradient computation via backprop, and this option prevents 
converting the torch tensor output by the model to a numpy array.
Using the torch.no_grad() context manager resolves this problem
as all tensors are set to requires_grad=False within the context manager.

An alternative to using the context manager is to do 
plt.plot(xx, model(xx).detach().clone())
The .detach().clone() operation create a copied pytorch tensor that
has requires_grad=False.

To be more precise, .detach() creates another tensor with requires_grad=False
(it is detached from the computation graph) but this tensor shares the same
underlying data with the original tensor. Therefore, this is not a genuine
copy (not a deep copy) and modifying the detached tensor will affect the 
original tensor is weird ways. The .clone() further proceeds to create a
genuine copy of the detached tensor, and one can freely manipulate and change it.
(For the purposes of plotting, it is fine to just call .detach() without
.clone() since plotting does not change the tensor.)

This discussion will likely not make sense to most students at this point of the course.
We will revisit this issue after we cover backpropagation.

'''
