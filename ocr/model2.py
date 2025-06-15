import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(#特征提取部分
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            #第一个参数 1：表示输入特征图的通道数。在这里，输入特征图是灰度图像，通道数为1，因为灰度图只有一个通道。
            #第二个参数 6：表示输出特征图的通道数。这个数字可以理解为卷积核的数量，也就是需要学习的滤波器的个数。每个滤波器将生成一个输出特征图，而这里有6个滤波器。
            #第三个参数 kernel_size=5：表示卷积核的大小。这里使用了一个5x5的卷积核，它会在输入特征图上进行滑动操作，提取局部特征。
            #第四个参数 padding=2：表示在输入特征图周围添加的填充数。填充可以保持输入和输出特征图的尺寸一致，避免尺寸缩小过快。具体来说，填充值是在输入特征图的边缘添加的像素数，这里填充值为2。
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, padding=0),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(#分类部分
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),#默认0.5
            nn.Linear(84, num_classes)#最后输出层输出维度为num_classes
        )

    def forward(self, x):#forward 方法定义了数据在模型中的前向传播过程
        x = self.features(x)#输入 x 通过特征提取部分 features 处理得到特征表示
        x = x.view(-1, 120)#将特征表示展平为一维向量（-1 表示自适应维度） 将张量 x 重新调整为一个新的形状，其中第一个维度的大小由张量总元素数和第二个参数确定，而第二个维度的大小为 120。
        x = self.classifier(x)#通过分类器部分 classifier 进行分类预测，返回预测结果。
        return x
