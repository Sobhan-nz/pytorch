# 🔥 مقدمه‌ای بر PyTorch — فریم‌ورک محبوب یادگیری عمیق

> "Flexible, fast, and deeply Pythonic."  
> — PyTorch.org

---

## ✅ PyTorch چیست؟

[**PyTorch**](https://pytorch.org/) یک فریم‌ورک متن‌باز برای یادگیری ماشین و یادگیری عمیق (Deep Learning) است که توسط Facebook AI توسعه داده شده. این کتابخانه، به دلیل سادگی در استفاده، انعطاف‌پذیری بالا و ترکیب عالی با اکوسیستم پایتون، به یکی از محبوب‌ترین ابزارهای توسعه مدل‌های هوش مصنوعی تبدیل شده است.

---

## 🚀 ویژگی‌های کلیدی PyTorch

- **ساختار پویا (Dynamic Computation Graph)**:  
  بر خلاف TensorFlow 1.x، گراف محاسباتی در PyTorch به‌صورت پویا ساخته می‌شود. یعنی در هر اجرا، گراف متفاوتی بر اساس کد شما تولید می‌شود — بسیار مناسب برای دیباگ و مدل‌های پیچیده.

- **یکپارچه با پایتون**:  
  PyTorch یک کتابخانه کاملاً پایتونی است. اگر بلد باشی پایتون بنویسی، با PyTorch احساس راحتی خواهی کرد.

- **پشتیبانی GPU با CUDA**:  
  تنها با `.to("cuda")` می‌تونی مدل و داده‌هات رو به کارت گرافیک منتقل کنی — بدون پیچیدگی.

- **کتابخانه torch.nn برای ساخت مدل‌ها**  
  شامل ابزارهایی برای تعریف شبکه‌های عصبی، لایه‌ها (مانند Linear، Conv2D)، توابع فعال‌سازی، loss function و غیره.

---

## 🧠 مثال ساده یک شبکه عصبی در PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# مدل ساده با یک لایه مخفی
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# داده فرضی
inputs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
targets = torch.tensor([[1.0], [0.0]])

# ساخت مدل
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# آموزش ساده
for epoch in range(100):
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training finished. Final loss:", loss.item())
