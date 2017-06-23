# 人脸自动分类主要步骤

## 1. 图像预处理

### 1.1. 主要步骤
1. 截取图像中人脸的部分（这个部分最好是方形的）。人脸部分截取包括`人脸校正`和`人脸剪裁`。
2. 对截取后的图像进行`几何归一化`。将截取的图像归一化为统一尺寸，便于后续处理，尺寸是一个变量，可以手动调节。
3. 对尺寸归一化后的图像进行`灰度归一化`。使用`直方图均衡化`的方法，增加图像对比度，减少光照对图像的影响。

### 1.2. 之前`matlab`处理步骤：
在对人脸图像的预处理中，人脸的归一化处理是至关重要的一环，它涉及到下一步处理的好坏。人脸的归一化包括`几何归一化`和`灰度归一化`:
* 几何归一化分两步：`人脸校正`和`人脸裁剪`。
* 灰度归一化主要是`增加图像的对比度`，`进行光照补偿`。


## `几何归一化`:  
几何归一化的目的主要是将人脸图像图像变换为统一的尺寸，有利于后期人脸特征的提取。
1. 标定特征点，这里用[x,y] = ginput(3)函数来标定两眼和鼻子三个特征点。主要是用鼠标动手标定，获取三个特征点的坐标值。
2. 根据左右两眼的坐标值旋转图像，以保证人脸方向的一致性。设两眼之间的距离为d，其中点为O。
3. 根据面部特征点和几何模型确定矩形特征区域，以O为基准，左右各剪切d，垂直方向各取0.5d和1.5d的矩形区域进行裁剪。
4. 对表情子区域图像进行尺度变换为统一的尺寸，更有利于表情特征的提取。把截取的图像统一规格为90*100的图像，实现图像的几何归一化。  

面部几何模型如下图：  
![面部几何模型](http://img.blog.csdn.net/20130709122421625?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY2hlbnl1MTk4ODAzMDI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## `灰度归一化`
灰度归一化:主要是增加图像的亮度，使图像的细节更加清楚，以减弱光线和光照强度的影响。
>`matlab`中用的是`image=255*imadjust(C/255,[0.3;1],[0;1]);` 用此函数进行光照补偿。 

```matlab
具体代码如下：

C= imread('Image001.jpg');
figure(1),imshow(C);
C=double(C);
image=255*imadjust(C/255,[0.3;1],[0;1]);
figure(2),imshow(image/255);
title('Lighting compensation');%光照补偿

[x,y] = ginput(3);    %%1 left eye, 2 right eye, 3 top of nose
cos = (x(2)-x(1))/sqrt((x(2)-x(1))^2+(y(2)-y(1))^2);
sin = (y(2)-y(1))/sqrt((x(2)-x(1))^2+(y(2)-y(1))^2);
mid_x = round((x(1)+x(2))/2);
mid_y = round((y(2)+y(1))/2);
d = round(sqrt((x(2)-x(1))^2+(y(2)-y(1))^2));
rotation = atan(sin./cos)*180/pi;
img = imrotate(image,rotation,'bilinear','crop'); 
figure(3), imshow(img);%人脸校正

[h,w] = size(img);
leftpad = mid_x-d;
if leftpad<1
   leftpad = 1;
end
toppad =mid_y - round(0.5*d);
if toppad<1
   toppad = 1;
 end
 rightpad = mid_x + d;
 if rightpad>w
    rightpad = w;
 end
 bottompad = mid_y + round(1.5*d);
 if bottompad>h
    bottompad = h;
 end   
 I1 =[];
 I2 =[];
 I1(:,:) = img(toppad:bottompad,leftpad:rightpad);
 I2(:,:) = imresize(I1,[90 100]); 
 figure(4),imshow(I2,[]);%人脸裁剪
```

>Python及c参考：    
`根据双眼的坐标对齐人脸Python实现`   
http://blog.csdn.net/haoji007/article/details/52775697  
`【计算机视觉】对检测的人脸进行剪切和归一化 `  
http://www.xuebuyuan.com/2225773.html   
`Implements face recognition algorithms for MATLAB/GNU Octave and Python.`   
https://github.com/bytefish/facerec  
http://www.bytefish.de/blog/fisherfaces/ 

## 2. 图像特征提取
### 2.1. 主要步骤
1. 训练脸部68特征点模型，前端返回标记的图片，后端返回68点坐标值，提取脸部几何特征。
2. 提取面部纹理特征。
3. 提取局部特征。

特征 | 提取方法
---|---
前额 | d(17,26)
眼距 |  d(39,42)/d(36,45)
内眦赘皮 | 纹理特征
鼻梁 | d(30,33)
面部黑痣 | 斑点检测

## 3. 自动分类
### 3.1. 主要步骤
1. 对提取的特征进行降维处理。
2. 将提取的特征与已经训练好的模型进行对比，得出结果。