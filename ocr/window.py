import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import os
import torch
import numpy as np
from torchvision import transforms
from new_model import LeNet5
#这里导入了所需的库，包括tkinter用于GUI界面，PIL用于图像处理，os用于文件操作，torch用于深度学习，numpy用于数值计算，transforms用于数据转换，以及自定义的LeNet5模型。

# 设置画板宽度、高度和画笔颜色，白色为轨迹颜色，黑色为底色
canvas_width = 400
canvas_height = 400
brush_size = 10
brush_color = 'white'
bg_color = 'black'

# 加载 LeNet-5 模型及其预训练参数 这里指定了模型路径、类别标签以及加载并初始化了LeNet5模型。
model_path = 'a.pth'
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model_dict = torch.load(model_path)
model = LeNet5()
model.load_state_dict(model_dict)

# 定义转换器将图像数据从PIL.Image转换为符合要求的张量格式以及数据预处理
to_tensor = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# 当鼠标按下并移动时，将轨迹绘制到画板上
def paint(event):
    x1, y1 = (event.x - brush_size), (event.y - brush_size)
    x2, y2 = (event.x + brush_size), (event.y + brush_size)
    canvas.create_oval(x1, y1, x2, y2, fill=brush_color, outline=brush_color)
    draw.line([x1, y1, x2, y2], fill='white', width=brush_size * 2)
#这个函数用于在画板上绘制轨迹。当用户按下鼠标并移动时，会在画板上绘制一个椭圆形的轨迹，同时使用ImageDraw绘制一条白色线条，实现了实时绘图的效果。

# 按预测按钮时执行的函数
def predict():#这个函数用于预测用户绘制的字母。首先将当前画布保存为一个临时PNG文件，然后加载该文件并转换为模型需要的张量格式。接着使用LeNet-5模型对输入图像进行预测，得到预测结果并显示在一个消息框中。
    # 以PNG格式保存当前画布
    filename = os.path.join('./', 'temp.png')
    image.save(filename)

    # 加载本地文件作为输入图像，并由转换器处理图像，以便进行分类
    input_image = Image.open(filename).convert('RGB')
    input_tensor = to_tensor(input_image)
    input_tensor = input_tensor.unsqueeze(0)

    # 使用先前训练的模型进行手写字母分类
    with torch.no_grad():
        output = model(input_tensor)
        print(output.numpy())
        predicted_class_index = output.numpy().argmax()
        predicted_class = classes[predicted_class_index]
        messagebox.showinfo("Prediction Result", f'The predicted letter is {predicted_class}.')


# 清除画板上的内容
def clear_canvas():
    canvas.delete('all')
    draw.rectangle((0, 0, canvas_width, canvas_height), fill=bg_color)
#这个函数用于清除画板上的内容。它会删除画板上的所有元素，并重新绘制一个背景色矩形，相当于清空了画板。

# 创建一个窗口，并添加名为 "Paint Board" 的画布和 "Predict" 和 "Clear" 两个按钮。
#创建了一个窗口和画布，以及两个按钮用于预测和清除操作。然后创建了一个Image实例和对应的ImageDraw对象，用于在画布上绘制图像。最后，绑定了鼠标事件和消息循环，使应用程序可以正常交互和工作。
root = tk.Tk()
root.title('Paint Board')

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg=bg_color)
canvas.pack()

predict_button = tk.Button(root, text='Predict', command=predict)
predict_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(root, text='Clear', command=clear_canvas)
clear_button.pack(side=tk.LEFT, padx=10)

# 使用 PIL.ImageDraw 创建一个可以在画板上绘图的实例
image = Image.new("RGB", (canvas_width, canvas_height), bg_color)
draw = ImageDraw.Draw(image)

canvas.bind('<B1-Motion>', paint)

root.mainloop()