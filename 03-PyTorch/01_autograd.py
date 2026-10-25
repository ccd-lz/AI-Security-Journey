import torch

# 定义模型参数和训练数据
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x = torch.tensor(3.0)
y = torch.tensor(7.0)

# 前向传播和计算损失
y_pred = w * x + b
loss = (y_pred - y) ** 2

# 反向传播，计算梯度
loss.backward()  # 不会直接修改参数数值

print("更新前w=", w.item())
print("更新前b=", b.item())
print("w的梯度=", w.grad.item())
print("b的梯度=", b.grad.item())

# 根据梯度更新参数
learning_rate = 0.01

with torch.no_grad():
    w -= learning_rate * w.grad  # 不会自动重新计算梯度
    b -= learning_rate * b.grad

print("更新后w=", w.item())
print("更新后b=", b.item())

w.grad.zero_()  # 下划线_表示原地操作，不会影响w的值
b.grad.zero_()  # 梯度不会自动清零，下一轮计算前先清除上一轮梯度

# 第二轮：参数更新后进行新一轮计算，观察清零后的新梯度
y_pred2 = w * x + b
loss2 = (y_pred2 - y) ** 2

print("第二轮预测值 =", y_pred2.item())
print("第二轮损失 =", loss2.item())

loss2.backward()

print("第二次 backward 后的 w.grad =", w.grad.item())
print("第二次 backward 后的 b.grad =", b.grad.item())
