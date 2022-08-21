import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 读取数据，并作初始化操作
train_data = pd.read_csv('./covid.train.csv').values
x_train, y_train = train_data[:, :-1], train_data[:, -1]


class COVID19Dataset(Dataset):
    """
    x: Features
    y: Targets, if none, do prediction
    """

    def __init__(self, x, y=None):  # Read data and prerocess
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):  # Return one sample at a time.
        # In this case, one sample includes a 117 dimensional feature and a label
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):  # Return the size of the dataset. In this case, it is 2699
        return len(self.x)


# 定义Dataset和Dataloader
train_dataset = COVID19Dataset(x_train, y_train)
# batch_size: 一次训练的样本数, shuffle: 打乱顺序, pin_memory: 锁页内存，不放在硬盘中，消耗内存资源但Tensor转义到GPU的显存会更快
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)


# 定义模型
class My_model(nn.Module):
    def __init__(self, input_dim):
        super(My_model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),  # 117->64
            nn.ReLU(),
            nn.Linear(64, 32),  # 64->32
            nn.ReLU(),
            nn.Linear(32, 1),  # 32->1
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x


# 模型，判断模型好坏的标准，优化器
# 定义模型并输入到GPU中
model = My_model(input_dim=x_train.shape[1]).to('cuda')
# 标准为均方误差
criterion = torch.nn.MSELoss(reduction='mean')
# 优化器为随机梯度下降(stochastic gradient descent), learn rate = 1e-5(移动的步长), momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

start_time = datetime.datetime.now()
# Training Loop
for epoch in range(3000):
    model.train()  # 设定模型为train mode
    train_pbar = tqdm(train_loader, position=0, leave=True)
    for x, y in train_pbar:
        x, y = x.to('cuda'), y.to('cuda')
        pred = model(x)  # 会调用forward函数进行模型训练
        loss = criterion(pred, y)  # 计算误差
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 梯度重置为0
end_time = datetime.datetime.now()
print("time consumed {}".format(end_time - start_time))
