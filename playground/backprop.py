import torch

x = torch.tensor(2.0)
y = torch.tensor(7.0)

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

prediction = w * x + b
loss = (prediction - y) ** 2

loss.backward()

print(w.grad)  # tensor(-16.)
print(b.grad)  # tensor(-8.)

with torch.no_grad():
    w -= 0.1 * w.grad
    b -= 0.1 * b.grad

w.grad.zero_()
b.grad.zero_()