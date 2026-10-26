import torch

# 初始化模型参数
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 两个训练样本
x = torch.tensor([1.0, 3.0])
y = torch.tensor([3.0, 7.0])

learning_rate = 0.05

for epoch in range(500):
    y_pred = w * x + b

    # 计算损失
    loss = ((y_pred - y) ** 2).mean()

    # 清零梯度
    w.grad = None
    b.grad = None

    # 反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    if (epoch +1) % 50 ==0:
        print(
            f"第 {epoch + 1} 轮："
            f"loss={loss.item():.4f}, "
            f"w={w.item():.4f}, "
            f"b={b.item():.4f}" #此处显示的是更新前的损失和更新后的参数
        )