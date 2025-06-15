import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from model2 import LeNet5

# 数据增强和归一化处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载测试集数据
test_dataset = ImageFolder(root='D:/EMNISTshouxiezimu/there/数据集2/val', transform=transform)# 加载测试集数据并应用数据预处理
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)# 创建测试集数据加载器，每批次包含 64 张图像，不重排数据shuffle=False

# 加载保存的模型并移动到GPU
model = LeNet5(num_classes=10) # 实例化 LeNet5 模型，共有 10 个类别
model.load_state_dict(torch.load('saved_models/lenet5_images_100次_数字.pth'))# 加载保存的模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 判断设备是 GPU 还是 CPU
model.to(device)# 将模型移至对应设备

# 对测试集进行预测
model.eval()# 设置模型为评估模式
with torch.no_grad():# 不计算梯度
    correct = 0# 记录正确预测的样本数
    total = 0# 记录总样本数
    for images, labels in test_loader:# 遍历测试集数据
        images = images.to(device)# 将图像数据移至设备
        labels = labels.to(device)# 将标签数据移至设备
        outputs = model(images)# 模型预测
        _, predicted = torch.max(outputs.data, 1)# 获取预测结果
        total += labels.size(0)# 统计样本总数
        #print(f'predicted: {predicted}  labels: {labels}')
        correct += (predicted == labels).sum().item()# 统计正确预测的样本数

    print(f'Test Accuracy: {100*correct/total:.2f}%')# 打印测试准确率