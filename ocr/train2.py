import os
import time
import torch#导入 PyTorch 深度学习框架。
import torch.nn as nn#导入 PyTorch 中的神经网络模块。
import torch.optim as optim#导入 PyTorch 中的优化器模块。
from torch.utils.data import DataLoader# 从 PyTorch 中导入 DataLoader 类，用于批处理数据。
from torchvision.datasets import ImageFolder#从 torchvision 中导入 ImageFolder 类，用于加载图像数据集。
from torchvision.transforms import transforms#从 torchvision 中导入 transforms 模块，用于数据转换操作。
from model2 import LeNet5

# 设置随机数种子
torch.manual_seed(123)#设置随机数种子为 123，以确保实验的可重复性。

# 数据增强和归一化处理 创建一个数据预处理管道 transform，包括将图像转为灰度、调整大小为 (28, 28)、转换为张量，并进行归一化处理。
transform = transforms.Compose([
    transforms.Grayscale(),#将图像转为灰度
    transforms.Resize((28, 28)),#调整大小为28*28
    transforms.ToTensor(),#转换为张量
    transforms.Normalize(mean=[0.5], std=[0.5])#进行归一化处理
])

# 加载训练集数据 使用 ImageFolder 类加载训练集数据，并通过 DataLoader 进行批处理，设置批大小为 64。
train_dataset = ImageFolder(root='D:/EMNISTshouxiezimu/there/数据集2/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)#64

# 加载验证集数据 使用 ImageFolder 类加载验证集数据，并通过 DataLoader 进行批处理，设置批大小为 64。
val_dataset = ImageFolder(root='D:/EMNISTshouxiezimu/there/数据集2/val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)##64

# 构建模型 创建一个 LeNet5 模型的实例，用于对包含 10 个类别的数据集进行分类。
model = LeNet5(num_classes=10)

# 设置损失函数和优化器 定义交叉熵损失函数 nn.CrossEntropyLoss() 和随机梯度下降（SGD）优化器 optim.SGD，学习率为 0.01，动量为 0.9。
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    print("cpu")
else:
    print("gpu")

model.to(device)# 将模型移至对应设备
num_epochs = 100
for epoch in range(num_epochs): # 循环每个训练轮次
    start_time = time.time()# 记录起始时间
    running_loss = 0.0# 初始化累计损失
    total = 0# 初始化样本总数
    correct = 0# 初始化预测正确的样本数
    for images, labels in train_loader:# 遍历训练集数据
        images = images.to(device)# 将图片数据移至设备
        labels = labels.to(device)# 将标签数据移至设备
        optimizer.zero_grad()# 梯度清零
        outputs = model(images)# 喂入模型得到输出
        loss = criterion(outputs, labels)# 计算损失
        loss.backward()# 反向传播求梯度
        optimizer.step()# 更新模型参数

        running_loss += loss.item() * images.size(0)# 累加损失
        _, predicted = torch.max(outputs.data, 1) # 获取预测结果
        total += labels.size(0)# 统计样本总数
        correct += (predicted == labels).sum().item()# 统计预测正确的样本数

    end_time = time.time()# 记录结束时间
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f}, Training Accuracy: {100*correct/total:.2f}%, Time: {end_time-start_time:.2f}s')



# 保存模型
if not os.path.exists('saved_models'): # 如果保存模型的目录不存在
    os.makedirs('saved_models')# 创建目录
torch.save(model.state_dict(), 'saved_models/lenet5_images_100次_数字.pth')# 将模型参数保存到文件中