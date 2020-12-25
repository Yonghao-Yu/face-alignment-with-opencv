import os
import math
import cv2 as cv
import numpy as np


def FcaeAlignment(img, landmarks):
    """
    function: face alignment
    img: 扣出来之后的人脸原图
    landmarks: 顺序：鼻子，左眼，右眼，左嘴，右嘴
    """
    # Step 1 get euler angle for filter
    size = img.shape
    # The order of key points must be consistent with model_points
    image_points = np.array([[landmarks[4], landmarks[5]],
                             [landmarks[0], landmarks[1]],
                             [landmarks[2], landmarks[3]],
                             [landmarks[6], landmarks[7]],
                             [landmarks[8], landmarks[9]]], dtype=np.double)
    
    model_points = np.array([(0.0, 0.0, 0.0),             # Nose tip
                             (-165.0, 170.0, -135.0),     # Left eye left corner
                             (165.0, 170.0, -135.0),      # Right eye right corner
                             (-150.0, -150.0, -125.0),    # Left Mouth corner
                             (150.0, -150.0, -125.0)])    # Right mouth corner
                        
    # Camera internals
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix,
                                                                 dist_coeffs, flags=1)
    axis = np.float32([[400, 0, 0],
                       [0, 400, 0],
                       [0, 0, 400]])
                          
    imgpts, jac = cv.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv.decomposeProjectionMatrix(proj_matrix)[6] 

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    # Step 2 face alignment
    if abs(yaw) > 40:
        retval = -1
        # return retval
    else:
        retval = 1
        center_x = (size[1]-1)//2
        center_y = (size[0]-1)//2
        M = np.float32([[1, 0, center_x - landmarks[4]],
                        [0, 1, center_y - landmarks[5]]])
        dst_temp = cv.warpAffine(img, M, (size[1], size[0]))

        dx = (landmarks[2] - landmarks[0])
        dy = (landmarks[3] - landmarks[1])
        angle = math.atan2(dy, dx) * 180. / math.pi
        M = cv.getRotationMatrix2D((center_x,center_y), angle, 1)
        dst = cv.warpAffine(dst_temp, M, (size[1], size[0]))

        '''
        (pitch, yaw, roll): 人脸角度，分别是点头、摇头和摆头。
        retval: 如果是 -1 判断左右摇头角度过大，是 1 则在正常范围内。
        dst: 校正后的人脸。
        '''
        return (pitch, yaw, roll), retval, dst


if __name__ == '__main__':

    face_dict = {'1': [353, 193, 398, 208, 362, 222, 340, 241, 379, 255],
                 '2': [204, 267, 290, 264, 247, 304, 209, 355, 289, 351],
                 '3': [182, 99, 214, 96, 200, 120, 185, 126, 214, 124],
                 '4': [159, 136, 173, 136, 169, 150, 153, 153, 164, 154],
                 '5': [95, 144, 105, 128, 109, 139, 115, 149, 126, 136]}

    file_path = '1.jpg'
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    face = face_dict[filename]
    
    img = cv.imread(file_path)

    # Draw the key points 
    cv.circle(img, (face[0], face[1]), 1, (255, 0, 0), 3)
    cv.circle(img, (face[2], face[3]), 1, (0, 255, 0), 3)
    cv.circle(img, (face[4], face[5]), 1, (0, 0, 255), 3)
    cv.circle(img, (face[6], face[7]), 1, (0, 0, 0), 3)
    cv.circle(img, (face[8], face[9]), 1, (255, 255, 255), 3)

    angle, retval, dst = FcaeAlignment(img, face)
    print(angle)
    print(retval)

    # if retval == -1:
    #     print('人脸倾斜角度过大，已被筛选！')
    # else:
    print('图片已被矫正！')
    cv.imshow("img", img)
    cv.imshow("dst", dst)
    cv.waitKey()
