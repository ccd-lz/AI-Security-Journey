import torch

# 初始化模型参数
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 训练数据
x = torch.tensor(3.0)
y = torch.tensor(7.0)

learning_rate = 0.01

# 进行十轮训练
for epoch in range(10):

    # 1. 前向传播
    y_pred = w * x + b

    # 2. 计算损失
    loss = (y_pred - y) ** 2

    # 3. 清空旧梯度；第一轮 grad 为 None，直接赋值也不会报错
    w.grad = None
    b.grad = None

    # 或者先判断梯度是否存在，再原地清零：
    # if w.grad is not None:
    #     w.grad.zero_()
    # if b.grad is not None:
    #     b.grad.zero_()

    # 4. 反向传播
    loss.backward()

    # 5. 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 6. 输出训练结果
    print(
        f"第 {epoch + 1} 轮："
        f"loss={loss.item():.4f}, "
        f"w={w.item():.4f}, "
        f"b={b.item():.4f}"
    )
