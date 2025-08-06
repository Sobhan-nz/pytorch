# 🔥 PyTorch - آشنایی با فریم‌ورک محبوب یادگیری عمیق

> “Flexible, fast, and deeply Pythonic.”  
> — PyTorch.org

---

## 🧠 PyTorch چیست؟

[**PyTorch**](https://pytorch.org/) یک فریم‌ورک متن‌باز یادگیری ماشین و یادگیری عمیق است که توسط Facebook AI توسعه داده شده. این کتابخانه با سادگی، انعطاف‌پذیری بالا، و هم‌خوانی کامل با پایتون، تبدیل به انتخاب اول بسیاری از توسعه‌دهندگان، محققان و علاقه‌مندان به AI شده است.

---

## 🚀 ویژگی‌های کلیدی PyTorch

- ✅ **گراف محاسباتی پویا (Dynamic Computation Graph)**
- ✅ **نوشتار کاملاً پایتونی و ساده**
- ✅ **پشتیبانی آسان از CUDA و GPU**
- ✅ **ساخت مدل‌های پیچیده با ماژول‌های torch.nn**
- ✅ **جامعه فعال، منابع گسترده و مدل‌های از پیش‌آموزش‌دیده**

---

## ⚡ مثال ساده از آموزش یک مدل

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# داده‌های فرضی
inputs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
targets = torch.tensor([[1.0], [0.0]])

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished. Final loss:", loss.item())
