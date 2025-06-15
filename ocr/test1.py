import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from new_model import LeNet5

# 数据增强和归一化处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载测试集数据
test_dataset = ImageFolder(root='D:/EMNISTshouxiezimu/there/images/val', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载保存的模型并移动到GPU
model = LeNet5(num_classes=26)
model.load_state_dict(torch.load('saved_models/lenet5_images_100次_新模型.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 对测试集进行预测
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #print(f'predicted: {predicted}  labels: {labels}')
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100*correct/total:.2f}%')