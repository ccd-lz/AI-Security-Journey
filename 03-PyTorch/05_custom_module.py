import torch
from torch import nn

# 采用自定义模块的方式实现线性回归模型
class MyLinearModel(nn.Module):
    # 初始化模型对象；创建 model = MyLinearModel() 时执行一次
    def __init__(self):
        super().__init__()  # 调用 nn.Module 的初始化逻辑，使子模块和参数能够被正确登记与管理
        self.linear = nn.Linear(1,1)  # 用实例属性持有可训练子模块；该层包含 weight 和 bias 参数

    # 定义前向计算；调用 model(x) 时由 nn.Module 的调用流程触发
    def forward(self, x):
        return self.linear(x)  # 使用已有线性层计算 y = wx + b，得到预测值

# 1. 训练样本
x = torch.tensor([[1.0], [3.0]])
y = torch.tensor([[3.0], [7.0]])

model = MyLinearModel() #创建自定义模型的一个对象

# 2. 创建损失函数
criterion = nn.MSELoss() #创建一个均方误差损失函数对象

# 3. 创建优化器
optimizer = torch.optim.SGD( #创建并配置一个优化器对象，其中model.parameters()：获取模型中登记的可训练参数；SGD：随机梯度下降优化器；lr:learning_rate,学习率
    model.parameters(),
    lr=0.05
)

# 5. 训练
for epoch in range(500):

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
            f"loss={loss.item():.6f}, "
            # model.linear 是已登记的子模块，weight 是其可训练参数；
            # item() 将单元素张量转为 Python 数值，:.6f 控制显示 6 位小数
            f"weight = {model.linear.weight.item():.6f}, "
            # bias 同样是 model.linear 的可训练参数，转换和显示格式与 weight 相同
            f"bias = {model.linear.bias.item():.6f}"
        )
