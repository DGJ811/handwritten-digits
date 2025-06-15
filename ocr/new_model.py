import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        #定义了模型的特征提取部分
        self.features = nn.Sequential(#self.features = nn.Sequential(...)：使用nn.Sequential定义了一个网络层序列，按顺序包含了一系列卷积层、激活函数和池化层。
            #每个卷积层包含了一个nn.Conv2d表示二维卷积操作，接着是一个nn.ReLU表示ReLU激活函数的应用。
            #池化层则有nn.AvgPool2d表示平均池化和nn.MaxPool2d表示最大池化，分别用于下采样和提取特征。
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))：定义了一个自适应平均池化层，将特征图的大小调整为(1, 1)，用于全局平均池化。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #定义了模型的分类器部分，包含了全连接层和一些附加层
        self.classifier = nn.Sequential(#self.classifier = nn.Sequential(...)：同样使用nn.Sequential定义了一个网络层序列，包括了展平层、全连接层、激活函数、Dropout层和最终的全连接输出层。
            nn.Flatten(),#nn.Flatten()用于将多维输入展平成一维向量。
            nn.Linear(64, 512),#nn.Linear()表示全连接层，接着是ReLU激活函数和Dropout层，用于防止过拟合。
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes)#最后一个全连接层的输出维度为num_classes，即分类的类别数。
        )

    def forward(self, x):#def forward(self, x):：定义了模型的前向传播函数，规定了数据在模型中的流动方式。
        #前向传播首先通过特征提取部分self.features提取特征，然后经过自适应平均池化层和分类器部分self.classifier进行分类预测，最终返回预测结果。
        x = self.features(x)
        print(x.size()) # 1 64 3 3
        x = self.avgpool(x)
        print(x.size()) # 1 64 1 1
        x = x.view(-1, 64)  # 将特征表示展平为一维向量（-1 表示自适应维度） 将张量 x 重新调整为一个新的形状，其中第一个维度的大小由张量总元素数和第二个参数确定，而第二个维度的大小为 120。
        print(x.size())  # 1 64
        x = self.classifier(x)
        print(x.size()) # 1 26
        print(x)
        return x