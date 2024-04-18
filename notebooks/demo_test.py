# -*- coding：utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/07/14; 15:19
# -*- python3.5
"""
address: https://blog.csdn.net/qq_35608277/article/details/78660469
本文代码来自于github中微软官方仓库
"""
import os
import cv2
import math
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
import matplotlib.image as mpimg
from notebooks import visualization
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
import sys

sys.path.append('./SSD_Tensorflow_master/')

slim = tf.contrib.slim
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

l_VOC_CLASS = ['ship']

net_shape = (300, 300)
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
data_format = 'NHWC'

image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format,
    resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)

image_4d = tf.expand_dims(image_pre, 0)

reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

ckpt_filename = '../train_model/model.ckpt-1486'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

ssd_anchors = ssd_net.anchors(net_shape)


def colors_subselect(colors, num_classes=1):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i * dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]


        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (l_VOC_CLASS[int(classes[i]) - 1], scores[i])
        p1 = (p1[0] - 5, p1[1])
        # cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)


colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=1)


# 主流程函数
def process_image(img, case, select_threshold=0.23, nms_threshold=.1, net_shape=(300, 300)):
    # select_threshold：box阈值——每个像素的box分类预测数据的得分会与box阈值比较，高于一个box阈值则认为这个box成功框到了一个对象
    # nms_threshold：重合度阈值——同一对象的两个框的重合度高于该阈值，则运行下面去重函数

    # 执行SSD模型，得到4维输入变量，分类预测，坐标预测，rbbox_img参数为最大检测范围，本文固定为[0,0,1,1]即全图
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions,
                                                               localisations, bbox_img], feed_dict={img_input: img})

    # ssd_bboxes_select()函数根据每个特征层的分类预测分数，归一化后的映射坐标，
    # ancohor_box的大小，通过设定一个阈值计算得到每个特征层检测到的对象以及其分类和坐标
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(rpredictions, rlocalisations, ssd_anchors,
                                                              select_threshold=select_threshold,
                                                              img_shape=net_shape,
                                                              num_classes=1, decode=True)

    """
    这个函数做的事情比较多，这里说的细致一些：
    首先是输入，输入的数据为每个特征层（一共6个，见上文）的：
                                                rpredictions: 分类预测数据，
                                                rlocalisations: 坐标预测数据，
                                                ssd_anchors: anchors_box数据
                                            其中：
                                               分类预测数据为当前特征层中每个像素的每个box的分类预测
                                               坐标预测数据为当前特征层中每个像素的每个box的坐标预测
                                               anchors_box数据为当前特征层中每个像素的每个box的修正数据
        函数根据坐标预测数据和anchors_box数据，计算得到每个像素的每个box的中心和长宽，这个中心坐标和长宽会根据一个算法进行些许的修正，
    从而得到一个更加准确的box坐标；修正的算法会在后文中详细解释，如果只是为了理解算法流程也可以不必深究这个，因为这个修正算法属于经验算
    法，并没有太多逻辑可循。
        修正完box和中心后，函数会计算每个像素的每个box的分类预测数据的得分，当这个分数高于一个阈值（这里是0.5）则认为这个box成功
    框到了一个对象，然后将这个box的坐标数据，所属分类和分类得分导出，从而得到：
        rclasses：所属分类
        rscores：分类得分
        rbboxes：坐标
        最后要注意的是，同一个目标可能会在不同的特征层都被检测到，并且他们的box坐标会有些许不同，这里并没有去掉重复的目标，而是在下文
    中专门用了一个函数来去重
    """

    # 检测有没有超出检测边缘
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    # 去重，将重复检测到的目标去掉
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # 将box的坐标重新映射到原图上（上文所有的坐标都进行了归一化，所以要逆操作一次）
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    if case == 1:
        bboxes_draw_on_img(img, rclasses, rscores, rbboxes, colors_plasma, thickness=8)
        return img
    else:
        return rclasses, rscores, rbboxes


"""
# 只做目标定位，不做预测分析
case = 1
img = cv2.imread("../demo/")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(process_image(img, case))
plt.show()
"""
# 2做目标定位，同时做预测分析
case = 1
path = '../demo/'
# 读取图片
from PIL import Image

image_names = sorted(os.listdir(path))
# 文件夹中的第几张图，-1代表最后一张
img = mpimg.imread(path + image_names[8])
rgb_img = np.expand_dims(img, axis=-1)  # 在最后一个维度添加一个维度，变为 256x256x1
rgb_img = np.tile(rgb_img, (1, 1, 3))  # 复制通道数为 3，变为 256x256x3


rclasses, rscores, rbboxes = process_image(rgb_img, case)
# visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
# 显示分类结果图
visualization.plt_bboxes(img, rclasses, rscores, rbboxes), rscores, rbboxes