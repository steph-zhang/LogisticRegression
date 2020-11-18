# author:zhang ming yi
import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1]], dtype=torch.float)


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_hat = torch.sigmoid(self.linear(x))
        return y_hat


model = LogisticModel()

criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10000):
    y_hat = model(x_data)
    loss_fuc = criterion(y_hat, y_data)
    print(epoch+1, loss_fuc.item())

    optimizer.zero_grad()
    loss_fuc.backward()
    optimizer.step()

x_test = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_hat = model(x_test)
print(y_hat)
