from PIL import Image
import os

def mirror_rotate_image(filename):
    # 打开PNG图片
    image = Image.open(filename)

    # 上下镜像反转
    mirrored_image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # 顺时针旋转90度
    rotated_image = mirrored_image.rotate(-90)

    # 构建保存的文件路径
    save_path = os.path.join(os.path.dirname(filename), "rotated_" + os.path.basename(filename))

    # 保存处理后的图片
    rotated_image.save(save_path)

if __name__ == "__main__":
    filename = r'D:/EMNISTshouxiezimu/there/数据集2/val/045890.png'  # 替换成你要处理的PNG图片文件名
    mirror_rotate_image(filename)
