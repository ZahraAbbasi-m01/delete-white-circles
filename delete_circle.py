import numpy as np
import cv2
import os


def delete_circle(image_path, dir_name, image_name):
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.imread(image_path)
    # make a mask to remove circle
    _, mask = cv2.threshold(cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY), 252, 255, cv2.THRESH_BINARY_INV)
    mod = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    erase = mask - mod
    dilation = cv2.dilate(erase, kernel, iterations=1)
    dst = cv2.inpaint(image, dilation, 3, cv2.INPAINT_TELEA)
    #save the result
    dir = os.path.join(r'D:\delete_circle_result', dir_name)
    os.makedirs(dir, exist_ok=True)
    image_name = os.path.join(dir, image_name)
    cv2.imwrite(image_name, dst)


for root, dirs, files in os.walk(r"E:\Datasets\share"):
    jpg_path = os.path.join(root, 'car_down_C.jpg')
    if os.path.exists(jpg_path):
        image_name = root.split("\\")[-1] + '.jpg'
        dir_name = root.split('_')[-2]
        delete_circle(jpg_path, dir_name, image_name)
