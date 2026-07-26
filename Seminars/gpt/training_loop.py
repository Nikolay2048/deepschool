import torch

x = torch.tensor([0.0, 1.0, 2.0, 3.0])
y = torch.tensor([1.0, 3.0, 5.0, 7.0])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.05

for epoch in range(100):
    # Forward pass
    prediction = w * x + b
    loss = ((prediction - y) ** 2).mean()

    # Backward pass
    loss.backward()

    # Обновление параметров
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Очистка градиентов
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0:
        print(
            f"epoch={epoch:3d}, "
            f"loss={loss.item():.6f}, "
            f"w={w.item():.4f}, "
            f"b={b.item():.4f}"
        )