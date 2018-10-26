import cv2
import glob
import time
import os
import datetime
import dlib
import numpy as np
import multiprocessing
from multiprocessing import Pool
import matplotlib.pylab as plt
import tqdm
import dataset
from mtcnn.mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PREDICTER_5_PATH = dataset.LANDMARKS_5_PATH
PREDICTER_68_PATH = dataset.LANDMARKS_68_PATH

def crop_rgb_face(img_path):
    # dlib人脸对齐
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTER_5_PATH)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(path))
        exit()

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(sp(img, detection))

    images = dlib.get_face_chips(img, faces, size=640, padding=0.25)

    # mtcnn人脸检测
    detector = MTCNN()

    image = images[0]

    result = detector.detect_faces(image)

    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']
    print(result)

    # 人脸部位剪裁
    crop_face = image[0:500,100:500]
    if bounding_box[1] < 0:
        bounding_box[3] = bounding_box[1]+bounding_box[3]
        bounding_box[1] = 0

    crop_face = image[bounding_box[1]:bounding_box[1]+bounding_box[3],bounding_box[0]:bounding_box[0]+bounding_box[2]]

    plt.figure(figsize=(15,15))
    plt.imshow(crop_face)

    crop_face = cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB)
    return crop_face

def face_pre_get_image_path(imagedir):
    return glob.glob(os.path.join(imagedir, '*'))

#几何归一化
def face_pre_zoom(img, width, height):
    zoom_face = cv2.resize(img, (height, width))
    return zoom_face

# 灰度归一化
def face_pre_gray(img):
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_face

# 直方图均衡化
def face_pre_histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    clahe_face = clahe.apply(img)
    return clahe_face


def faces_main(imagepath, imagedir, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    try:
        fs_align = crop_rgb_face(imagepath)
        # f_z = face_pre_zoom(fs_align, 320, 320)
        time.sleep(0.1)
        f_g = face_pre_gray(fs_align)
        clahe_face = face_pre_histogram_equalization(f_g)
        filepath, filename = os.path.split(imagepath)
        name, ext = os.path.splitext(filename)
        spath = os.path.join(savedir, name + '_%.f.jpg' % time.time())
        cv2.imwrite(spath, clahe_face)
        # print('                                           ')
    except Exception as e:
        print(e)

def main(maxprocess, function, imagedir, savedir):
    imgspath = face_pre_get_image_path(imagedir)

    p = Pool(maxprocess)
    for imgpath in tqdm.tqdm(imgspath):
        p.apply_async(function, args=(imgpath, imagedir, savedir,))
    p.close()# 关闭进程池
    p.join()# 等待所有子进程完毕

if __name__ == '__main__':
    imagedir = r'F:\facial_images\TS\TS_frontal'
    savedir = r'F:\facial_images\TS\TS1'

    main(4, faces_main, imagedir=imagedir, savedir=savedir)