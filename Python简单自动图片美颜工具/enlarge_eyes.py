

import cv2
import math
import numpy as np

left_pos_start = 33 # 左眼起始眼角
left_pos_end = 133 # 左眼终止眼角
right_pos_start = 263 # 右眼起始眼角
right_pos_end = 362 # 右眼终止眼角

def enlarge_eyes(enlarge_eyes_src, landmarks, image_width, image_height):
    left_face_start_x = landmarks[left_pos_start].x * image_width
    left_face_start_y = landmarks[left_pos_start].y * image_height
    left_face_end_x = landmarks[left_pos_end].x * image_width
    left_face_end_y = landmarks[left_pos_end].y * image_height
    right_face_start_x = landmarks[right_pos_start].x * image_width
    right_face_start_y = landmarks[right_pos_start].y * image_height
    right_face_end_x = landmarks[right_pos_end].x * image_width
    right_face_end_y = landmarks[right_pos_end].y * image_height
    PointX_left = int((left_face_start_x+left_face_end_x)/2)
    PointY_left = int((left_face_start_y+left_face_end_y)/2)
    PointX_right = int((right_face_start_x+right_face_end_x)/2)
    PointY_right = int((right_face_start_y+right_face_end_y)/2)
    Radius_left = int(math.sqrt(math.pow(left_face_start_x-left_face_end_x, 2) + math.pow(left_face_start_y-left_face_end_y, 2)))
    Radius_right = int(math.sqrt(math.pow(right_face_start_x-right_face_end_x, 2) + math.pow(right_face_start_y-right_face_end_y, 2)))
    #放大左眼
    enlarge_eyes_dst = enlarge_a_eye(enlarge_eyes_src, PointX_left, PointY_left, Radius_left, 19.78)
    #放大右眼
    enlarge_eyes_dst = enlarge_a_eye(enlarge_eyes_dst, PointX_right, PointY_right, Radius_right, 19.78)
    return enlarge_eyes_dst


def enlarge_a_eye(src, PointX, PointY, Radius, Strength):
    processed_image = np.zeros(src.shape, np.uint8)
    processed_image = src.copy()
    height = src.shape[0]
    width = src.shape[1]
    PowRadius = Radius * Radius
 
    maskImg = np.zeros(src.shape[:2], np.uint8)
    cv2.circle(maskImg, (PointX, PointY), math.ceil(Radius), (255, 255, 255), -1)
 
    mapX = np.vstack([np.arange(width).astype(np.float32).reshape(1, -1)] * height)
    mapY = np.hstack([np.arange(height).astype(np.float32).reshape(-1, 1)] * width)
 
    OffsetX = mapX - PointX
    OffsetY = mapY - PointY
    XY = OffsetX * OffsetX + OffsetY * OffsetY
 
    ScaleFactor = 1 - XY / PowRadius
    ScaleFactor = 1 - Strength / 100 * ScaleFactor
    UX = OffsetX * ScaleFactor + PointX
    UY = OffsetY * ScaleFactor + PointY
    UX[UX < 0] = 0
    UX[UX >= width] = width - 1
    UY[UY < 0] = 0
    UY[UY >= height] = height - 1
 
    np.copyto(UX, mapX, where=maskImg == 0)
    np.copyto(UY, mapY, where=maskImg == 0)
 
    UX = UX.astype(np.float32)
    UY = UY.astype(np.float32)
 
    processed_image = cv2.remap(src, UX, UY, interpolation=cv2.INTER_LINEAR)
    return processed_image



    



