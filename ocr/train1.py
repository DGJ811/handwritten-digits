import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from new_model import LeNet5

# 设置随机数种子
torch.manual_seed(123)

# 数据增强和归一化处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载训练集数据
train_dataset = ImageFolder(root='D:/EMNISTshouxiezimu/there/images/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载验证集数据
val_dataset = ImageFolder(root='D:/EMNISTshouxiezimu/there/images/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 构建模型
model = LeNet5(num_classes=26)

# 设置损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print("cpu")
else:
    print("gpu")

model.to(device)
num_epochs = 100
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0.0
    total = 0
    correct = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f}, Training Accuracy: {100*correct/total:.2f}%, Time: {end_time-start_time:.2f}s')



# 保存模型
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
torch.save(model.state_dict(), 'saved_models/lenet5_images_100次_新模型.pth')
