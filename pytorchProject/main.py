import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, file):
        self.data = 1


file = 1
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

x = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)
y = torch.zeros([2, 2])
z = torch.ones([1, 2, 5])  # 1x2x5的矩阵，数值为1

z = z + z
k = x - y
k = x.sum()
k = x.mean()
y = x.pow(2)
print(z.shape)
z = z.transpose(0, 1)  # 维度互换
print(z.shape)
z = z.squeeze(1)  # 移除等于1的维度，可指定
print(z.shape)
z = z.unsqueeze(2)  # 在指定位置添加一维
print(z.shape)

x = torch.zeros([2, 1, 3])
y = torch.zeros([2, 3, 3])
z = torch.zeros([2, 2, 3])
w = torch.cat([x, y, z], dim=1)
print(w.shape)

print(torch.cuda.is_available())
print(torch.__version__)

x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
print(x.grad)
k = x.pow(3).sum()
print(x.grad)
k.backward()
print(x.grad)

layer = torch.nn.Linear(32, 64)
print(layer.weight.shape)
print(layer.bias.shape)
torch.nn.Sigmoid()
torch.nn.ReLU()

import torch.nn as nn


class MyModel(nn.Module):
    def __int__(self):  # initialize your model and define layers
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )

    def forward(self, x):  # compute output of your NN
        return self.net(x)


criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

import torch
n_gpu = torch.cuda.device_count()
print(n_gpu)

print("使用gpu数量为：", torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
try:
    print(torch.cuda.get_device_name(1))
except:
    print("并未使用其他显卡")
