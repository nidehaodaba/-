import math
import numpy as np

left_pos_start = 132 # 左脸轮廓起始点
left_pos_end = 172 # 左脸轮廓终止点
right_pos_start = 361 # 右脸轮廓起始点
right_pos_end = 397 # 右脸轮廓终止点
common_end = 5 # 脸部中心，鼻尖处

def BilinearInsert(src,ux,uy):
    w,h,c = src.shape
    if c == 3:
        x1=int(ux)
        x2=x1+1
        y1=int(uy)
        y2=y1+1

    part1=src[y1,x1].astype(np.float)*(float(x2)-ux)*(float(y2)-uy)
    part2=src[y1,x2].astype(np.float)*(ux-float(x1))*(float(y2)-uy)
    part3=src[y2,x1].astype(np.float) * (float(x2) - ux)*(uy-float(y1))
    part4 = src[y2,x2].astype(np.float) * (ux-float(x1)) * (uy - float(y1))

    insertValue=part1+part2+part3+part4

    return insertValue.astype(np.int8)


def localTranslationWarp(srcImg,startX,startY,endX,endY,radius):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # 计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            #计算该点是否在形变圆的范围之内
            #优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i-startX)>radius and math.fabs(j-startY)>radius:
                continue

            distance = ( i - startX ) * ( i - startX) + ( j - startY ) * ( j - startY )

            if(distance < ddradius):
                #计算出（i,j）坐标的原坐标
                #计算公式中右边平方号里的部分
                ratio=(  ddradius-distance ) / ( ddradius - distance + ddmc)
                ratio = ratio * ratio

                #映射原位置
                UX = i - ratio  * ( endX - startX )
                UY = j - ratio  * ( endY - startY )

                #根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg,UX,UY)
                #改变当前 i ，j的值
                copyImg[j,i] =value
    return copyImg


def thin_face(thin_face_src, landmarks, image_width, image_height):
    left_face_start_x = landmarks[left_pos_start].x * image_width
    left_face_start_y = landmarks[left_pos_start].y * image_height
    left_face_end_x = landmarks[left_pos_end].x * image_width
    left_face_end_y = landmarks[left_pos_end].y * image_height
    right_face_start_x = landmarks[right_pos_start].x * image_width
    right_face_start_y = landmarks[right_pos_start].y * image_height
    right_face_end_x = landmarks[right_pos_end].x * image_width
    right_face_end_y = landmarks[right_pos_end].y * image_height
    common_end_x = landmarks[common_end].x * image_width
    common_end_y = landmarks[common_end].y * image_height

    thin_r_left = math.sqrt(math.pow(left_face_start_x-left_face_end_x, 2) + math.pow(left_face_start_y-left_face_end_y, 2))
    thin_r_right = math.sqrt(math.pow(right_face_start_x-right_face_end_x, 2) + math.pow(right_face_start_y-right_face_end_y, 2))

    #瘦左边脸
    thin_face_dst = localTranslationWarp(thin_face_src, left_face_start_x, left_face_start_y,  common_end_x, common_end_y, thin_r_left)
    #瘦右边脸
    thin_face_dst = localTranslationWarp(thin_face_dst, right_face_start_x, right_face_start_y,  common_end_x, common_end_y, thin_r_right)
    return thin_face_dst