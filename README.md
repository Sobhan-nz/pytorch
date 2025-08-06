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
```


## 🧰 ابزارهای مهم در اکوسیستم PyTorch

| کتابخانه             | توضیح                                                                 |
|----------------------|------------------------------------------------------------------------|
| `torchvision`        | شامل دیتاست‌ها، مدل‌های آماده و توابع مرتبط با بینایی کامپیوتر       |
| `torchaudio`         | برای پردازش صوت و گفتار                                                |
| `torchtext`          | مخصوص NLP و پردازش زبان طبیعی                                         |
| `PyTorch Lightning`  | ساده‌سازی کدنویسی مدل‌ها، کاهش پیچیدگی کدهای آموزشی و تولیدی         |
| `fastai`             | فریمورکی سطح بالا روی PyTorch برای آموزش سریع‌تر با API ساده‌تر     |


## 📚 منابع یادگیری بیشتر

- 🌐 [PyTorch Official Website](https://pytorch.org/)  
  وب‌سایت رسمی پروژه PyTorch با دسترسی به داکیومنت‌ها، آموزش‌ها و ابزارها

- 📄 [Documentation](https://pytorch.org/docs/)  
  مستندات رسمی کامل برای نسخه‌های مختلف PyTorch

- 🧪 [Tutorials](https://pytorch.org/tutorials/)  
  آموزش‌های گام‌به‌گام برای مباحث پایه تا پیشرفته

- 📘 [کتاب رایگان: Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)  
  کتاب رسمی PyTorch برای یادگیری عملی و پروژه‌محور یادگیری عمیق

- 🎓 دوره‌های رایگان در [Coursera](https://www.coursera.org/), [Udacity](https://www.udacity.com/), [YouTube](https://www.youtube.com/results?search_query=pytorch+tutorial) و سایر پلتفرم‌ها  
  آموزش‌های ویدیویی معتبر و رایگان برای یادگیری PyTorch


## 💬 چرا PyTorch؟

- 🧠 تقریباً تمام مقالات هوش مصنوعی مدرن از **PyTorch** استفاده می‌کنند.
- 🚀 اگر قصد آموزش، تحقیق یا ساخت پروژه‌های واقعی AI را داری، **PyTorch بهترین نقطه‌ی شروع** است.
- 🔬 حتی گوگل نیز در برخی پروژه‌های تحقیقاتی خود از PyTorch استفاده می‌کند — با وجود داشتن TensorFlow!

---

## 📎 درباره این پروژه

این مخزن با هدف معرفی اولیه PyTorch و ارائه‌ی مثال‌های ساده برای شروع یادگیری طراحی شده است.  
اگر برات مفید بود:

- ⭐ پروژه را ستاره‌دار کن
- 🗨 نظرت را در بخش [Issues](https://github.com/Sobhan-nz/pytorch-intro/issues) یا Discussions بنویس
- 🍴 فورک کن و آن را به سلیقه‌ی خودت ارتقا بده

---

📌 **نویسنده:** Sobhan Noorzahi  
📆 **تاریخ انتشار:** 2025-08-06  
📁 **پروژه آموزشی:** [github.com/your-username/pytorch-intro](https://github.com/Sobhan-nz/pytorch-intro)
