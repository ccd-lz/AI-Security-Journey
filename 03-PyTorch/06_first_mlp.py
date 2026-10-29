import torch
from torch import nn

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(1,4) # 输入层到隐藏层的线性变换，输入维度为1，输出维度为4

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(4,1) # 隐藏层到输出层的线性变换，输入维度为4，输出维度为1

    def forward(self,x):
        h = self.linear1(x) # 输入层到隐藏层的线性变换
        h = self.relu(h) # 激活函数ReLU
        y = self.linear2(h) # 隐藏层到输出层的线性变换
        return y

x = torch.tensor([[1.0], [3.0]])
model = MyMLP()

y_pred = model(x) # 前向传播，得到预测值

print("输入形状：", x.shape)
print("输出形状：", y_pred.shape)
print(model)

for name, param in model.named_parameters():
    print(name, param.shape, param.numel())

print("参数总数：", sum(p.numel() for p in model.parameters()))