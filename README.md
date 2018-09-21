# 特纳综合征（Turner syndrome）人脸自动分类系统
[![版本](https://img.shields.io/badge/python-2.7.12-blue.svg)](https://github.com/doraemonlei/faceDig)
[![系统](https://img.shields.io/badge/system-CentOS%2CUbuntu%2CWindows-blue.svg)](https://github.com/doraemonlei/faceDig)
[![谢谢](https://img.shields.io/badge/Say%20Thanks-!-green.svg)](https://github.com/doraemonlei/faceDig)

## 简介
由于个体临床症状的多样性以及缺乏可靠的诊断标准，对特纳综合征（TS）的初步诊断主要依靠身高这一临床特征，从而导致很多特纳综合征患者前期被诊断为矮小症等其它疾病。为了提高诊断客观性和降低医生的工作强度，同时为了能让患者早发现早治疗，造福患者，本研究提出了一种辅助诊断的自动人脸分类方法。  

本系统使用Python完成，大部分为函数式编程，进行了功能的简单实现，没有进行异常处理。系统中对人脸图像预处理，特征提取，自动分类预测等主要步骤实现自动处理，部分细节及功能还待完善。

### 目录结构

    │  .gitignore
    │  README.md
    │  requirements.txt
    ├─causalAnalysis 因果分析，显示局部特征对分类的影响程度
    │      show_degree.py
    ├─classification 自动分类预测
    │      local_adaboost.py
    ├─Documentation 文档
    │      main_steps.md
    ├─featureExtraction 特征提取
    │      global_geometrical_feature.py
    │      global_texture_feature.py
    │      local_feature.py
    ├─preprocessing 人脸图像预处理
    │      pre.py
    │      pre_1_normalization.py
    │      pre_2_zoom.py
    │      pre_3_gray.py
    │      pre_4_histograms.py
    │      pre_filter.py
    │      pre_saltandpepper.py
    │      pre_video.py
    └─tools 工具
        │  facedata.csv 训练数据
        │  tool_blob_detection.py 黑痣提取
        │  tool_crop_faceArea.py 脸部局部区域提取
        │  tool_csv.py 数据的csv处理
        │  tool_excel_openpyxl.py 数据excel处理
        │  tool_facedata.py 数据格式化
        │  tool_face_detection_bydlib.py 人脸检测dlib
        │  tool_face_detection_byopencv.py 人脸检测opencv
        │  tool_gabor_gui.py gabor参数设置opencv
        │  tool_gabor_skimage.py gabor滤波skimage
        │  tool_get_gabor_feature.py 获取gabor特征
        │  tool_get_grey_image.py 获取灰度人脸图像
        │  tool_get_landmarksAndImages.py 获取人脸特征点
        │  tool_pca_withnumpy.py pca特征降维
        │  tool_test_ROI.py 测试提取roi
        │  __init__.py
        │
        └─haarcascades opencv模型

## 部署环境
### 安装需求
* dlib==19.2.0 人脸识别，提取特征点
* opencv==3.2 人脸识别，图像处理，特征提取
* numpy==1.11.1+mkl 数据操作
* openpyxl==2.4.1 数据操作
* scikit-image==0.12.3 图像处理
* scikit-learn==0.18.1 分类预测

## 系统流程
* 人脸图像预处理(preprocessing)
    * 人脸识别，人脸校正，人脸提取:识别出给定图像中的人脸，根据双眼位置对人脸进行旋转校正，提取出校正后的人脸（本系统中提取的图像为方形）。
    * 人脸图像几何归一化：将提取的图像归一化为相同的尺寸。
    * 人脸图像灰度归一化：将几何归一化后的图像进行灰度归一化，转为灰度图像。
    * 人脸图像直方图均衡化：对灰度图像进行直方图均衡化，减少光照的影响。

* 特征提取(featureExtraction)
    * 全局几何特征提取（global_geometrical_feature.py），提取人脸的68个特征点，计算各点的距离，得到2278维的几何特征。
    * 全局纹理特征提取（global_texture_feature.py），将一张人脸图像分割成64份，对图像使用gabor滤波器组进行滤波后，分别提取每一块的能量，得到能量纹理特征矩阵。
    * 局部特征提取（local_feature.py），提取5个局部特征。方法如下：

                特征 | 提取方法
                ---|---
                前额 | d(17,26)
                眼距 |  d(39,42)/d(36,45)
                内眦赘皮 | 纹理特征
                鼻梁 | d(30,33)
                面部黑痣 | 斑点检测

* 自动分类(classification)
    * Svm 对于全局特征使用了libsvm进行分类与预测，精确度较低。
    * Adaboost（local_adaboost.py）对局部特征使用Adaboost算法进行分类与预测，精确度较高。get_predictf返回预测结果，0代表未患病，1代表患病。get_probaf返回对于预测患病与不患病的概率（[不患病的概率，患病的概率]）

* 因果分析(causalAnalysis)  
主要用来分析脸部5个特征对于分类的影响程度。  
颜色与数值对应（关联性从小到大）：
    * 绿色-->0
    * 蓝色-->1
    * 黄色-->2
    * 橙色-->3
    * 红色-->4

## 使用方法
三个步骤为分步执行，如需一次执行将三步写入一个脚本中即可。
### 预处理部分
pre.py->main(图片路径,保存文件的文件夹)，执行完成后将预处理后的图像进行保存
### 特征提取部分
local_feature.py->main(预处理完成的图像路径)，执行完成后返回特征列表([[图像1],[图像2]...])  
local_feature.py->main2(预处理完成的图像路径)，执行完成后返回一张图像的特征列表
### 分类识别
local_adaboost.py->get_predict(上一步提取的特征数据，迭代次数，步长)，返回预测值  
local_adaboost.py->get_proba(上一步特征数据，迭代次数，步长)，返回预测0和1的可能性值
### 因果分析
show_degree.py->show_degree(图像路径, 使用local_feature.py->main2获得的人脸特征)，返回分析后的图像
