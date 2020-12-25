# 介绍

用opencv去预测人脸旋转角度（给定人脸关键点）

# 文件夹

根目录下一共有两个文件：**face_alignment** 和 **camera_matrix**
	
## face_alignment
	
这个文件夹里面包括一个 python 文件和 5 张测试用例，python 文件实现了人脸倾斜角度预测和矫正，开发人员可自行验证

## camera_matrix

人脸角度预测用到了 solvePnP() 函数，需要传入相机内参矩阵和畸变矩阵，使用该文件夹内代码可以计算出这两个矩阵
	
文件夹内的图片是用我们自己的摄像头拍摄的，到时需要开发人员用自己的摄像头自行拍摄 10 到 20 张即可

如果只是测试图片而不是从摄像头实时获取，使用默认值即可（程序里有写）

# 效果展示

## before

![before](face_alignment/before.jpg)

## after

![after](face_alignment/after.jpg)

**有一个角度始终不太准，准确的话应该还是需要用深度学习的方式**