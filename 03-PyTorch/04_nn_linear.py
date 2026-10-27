import torch
from torch import nn

# 1. 两个训练样本
x = torch.tensor([[1.0], [3.0]])
y = torch.tensor([[3.0], [7.0]])

# 2. 创建线性模型
model = nn.Linear(1, 1) #表示每个样本输入一个特征，输出一个预测值

# 3. 创建损失函数
criterion = nn.MSELoss() #创建一个均方误差损失函数对象

# 4. 创建优化器
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
            f"w={model.weight.item():.4f}, "
            f"b={model.bias.item():.4f}"
        )
