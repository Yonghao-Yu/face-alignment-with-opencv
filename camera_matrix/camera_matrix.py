import glob
import cv2
import numpy as np

# 定义角点个数，我这里使用的图片有4*6=24个
cbraw = 6
cbcol = 4

# 准备世界坐标，24个三维坐标
objp = np.zeros((cbraw*cbcol,3), np.float32) 
'''
设定世界坐标下点的坐标值，因为用的是棋盘可以直接按网格取；
假定棋盘正好在x-y平面上，这样z值直接取0，简化初始化步骤。
mgrid把列向量[0:cbraw]复制了cbcol列，把行向量[0:cbcol]复制了cbraw行。
转置reshape后，每行都是4×6网格中的某个点的坐标。
'''
objp[:,:2] = np.mgrid[0:cbraw,0:cbcol].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# glob是个文件名管理工具
images = glob.glob("C:/Users/Administrator/Desktop/video/*.jpg")
for fname in images:
# 对每张图片，识别出角点，记录世界物体坐标和图像坐标
    img = cv2.imread(fname) #source image
    # img = cv2.resize(img,(224,224))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转灰度
    cv2.imshow('img',gray)
    cv2.waitKey(25)
    # 寻找角点，存入corners，ret是找到角点的flag
    ret, corners = cv2.findChessboardCorners(gray,(6,4),None)
    # criteria:角点精准化迭代过程的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 执行亚像素级角点检测
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

    objpoints.append(objp)
    imgpoints.append(corners2)
    # 在棋盘上绘制角点,只是可视化工具
    img = cv2.drawChessboardCorners(gray,(6,4),corners2,ret)
    cv2.imshow('img',img)
    cv2.waitKey(25)
'''
传入所有图片各自角点的三维、二维坐标，相机标定。
每张图片都有自己的旋转和平移矩阵，但是相机内参和畸变系数只有一组。
mtx，相机内参；dist，畸变系数；revcs，旋转矩阵；tvecs，平移矩阵。
'''
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
img = cv2.imread('C:/Users/Administrator/Desktop/video/1.jpg')
# img = cv2.resize(img,(224,224))
h,w = img.shape[:2]
'''
优化相机内参（camera matrix），这一步可选。
参数1表示保留所有像素点，同时可能引入黑色像素，
设为0表示尽可能裁剪不想要的像素，这是个scale，0-1都可以取。
'''
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# 纠正畸变
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# 这步只是输出纠正畸变以后的图片
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
# 打印我们要求的两个矩阵参数
print ("newcameramtx:\n",newcameramtx)
print ("dist:\n",dist)
# 计算误差
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print ("total error: ", tot_error/len(objpoints))