import torch
from torch import nn

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(1,16) # 输入层到隐藏层的线性变换，输入维度为1，输出维度为16

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(16,1) # 隐藏层到输出层的线性变换，输入维度为16，输出维度为1

    def forward(self,x):
        h = self.linear1(x) # 输入层到隐藏层的线性变换
        h = self.relu(h) # 激活函数ReLU
        y = self.linear2(h) # 隐藏层到输出层的线性变换
        return y

x = torch.linspace(-2, 2, 21).unsqueeze(1)# 创建一个一维张量，包含从-2到2的21个均匀间隔的点，并在第1个维度插入一个长度为1的维度,将其形状调整为(21, 1),目的是使数据符合nn.Linear的批量输入格式
y = x ** 2

model = MyMLP()
criterion = nn.MSELoss() #创建一个均方误差损失函数对象

optimizer = torch.optim.SGD( #创建并配置一个优化器对象，其中model.parameters()：获取模型中登记的可训练参数；SGD：随机梯度下降优化器；lr:learning_rate,学习率
    model.parameters(),
    lr=0.01
)

# 训练
for epoch in range(1000):

    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = criterion(y_pred, y) #调用criterion对象，相当于loss = ((y_pred - y) ** 2).mean()

    # 清理上一轮梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 使用优化器更新参数
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"第 {epoch + 1} 轮："
            f"loss={loss.item():.6f}"
        )

# 训练结束后，查看模型的预测效果
x_test = torch.tensor([[-1.9], [-1.3], [-0.7], [0.5], [1.7]])

model.eval() # 切换到评估模式；影响 Dropout、BatchNorm 等层的行为，但不会自动关闭梯度计算    

with torch.no_grad(): # 在评估模式下，不需要计算梯度，节省内存和计算资源
    y_test_prd = model(x_test)
    print("测试输入：", x_test.squeeze().tolist()) #.squeeze()：将张量中长度为1的维度去掉，得到一维张量
    print("测试输出：", y_test_prd.squeeze().tolist()) #.tolist()：将张量中的多个元素转换成 Python 列表。(另外：model.linear2.weight.detach().flatten().tolist()：.detach()：得到一个不再参与当前自动求导图的张量，用于观察参数的值，避免梯度计算；.flatten()：将张量展平为一维张量)