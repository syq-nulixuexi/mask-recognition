# Mask recognition
 This is a python program for mask recognition based on mobilenet v2
# 项目简介

这是一个基于mobilenet v2的python程序，主要用于识别电脑摄像头中捕获到的人是否佩戴口罩。

# 环境
python 3.6
opencv-python 4.4.0
tensorflow 2.1.0

# 项目文件

## model_v2.py

modilenet v2模型文件。

## train_mobilenet_v2.py

训练模型文件，训练集采用网上下载的mask与unmask小数据集图片。

## predict one picture.py

使用训练后的权重文件来预测一张图片为mask还是unmask，图片上方显示预测结果。


## predict.py

通过opencv使用电脑摄像头获取图片，将实时获取的图片传入模型中进行实时预测，图片上方显示帧率、预测结果。

## spilt_data.py

将训练的图片进行分类并贴标签。

# 遇到的问题及解决方法

opencv获取到的图片直接传入模型中会出现类型不匹配问题，使用Image.fromarray函数，将图片从array转换到image。


# 其他

目前项目识别率还较低，可能存在的问题：

1.训练使用的数据集数据量较少

2.训练集中图片背景单一，但实际测试时环境背景比较复杂

3.模型中参数有些没有进行优化
