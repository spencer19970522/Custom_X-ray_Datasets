# -*- coding: utf-8 -*-
import dota_utils as util
import os
import numpy as np
from PIL import Image
import cv2
import random
import  shutil
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
import time
import argparse

def dota2Darknet(imgpath, txtpath, dstpath, extractclassname):
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)
    os.makedirs(dstpath)
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        objects = util.parse_dota_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + '.png')
        img = Image.open(img_fullname)
        img_w, img_h = img.size

        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])
                else:
                    continue
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
                f_out.write(outline + '\n')

def dota2LongSideFormat(imgpath, txtpath, dstpath, extractclassname):
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)
    os.makedirs(dstpath)
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        objects = util.parse_dota_poly(fullname)

        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + '.png')
        img = Image.open(img_fullname)
        img_w, img_h = img.size

        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            num_gt = 0
            for i, obj in enumerate(objects):
                num_gt = num_gt + 1
                poly = obj['poly']
                poly = np.float32(np.array(poly))
                poly[:, 0] = poly[:, 0]/img_w
                poly[:, 1] = poly[:, 1]/img_h

                rect = cv2.minAreaRect(poly)
                #box = np.float32(cv2.boxPoints(rect))

                c_x = rect[0][0]
                c_y = rect[0][1]
                w = rect[1][0]
                h = rect[1][1]
                theta = rect[-1]  # Range for angle is [-90，0)

                trans_data = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                if not trans_data:
                    if theta != 90:
                        print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    num_gt = num_gt - 1
                    continue
                else:
                    c_x, c_y, longside, shortside, theta_longside = trans_data # range:[-180，0)

                bbox = np.array((c_x, c_y, longside, shortside))

                if (sum(bbox <= 0) + sum(bbox[:2] >= 1) ) >= 1:  # 0<xy<1, 0<side<=1
                    print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (c_x, c_y, longside, shortside, theta_longside))
                    num_gt = num_gt - 1
                    continue
                if (obj['name'] in extractclassname):
                    id = extractclassname.index(obj['name'])
                else:
                    print('预定类别中没有类别:%s;已将该box排除,问题出现在该图片中:%s' % (obj['name'], fullname))
                    num_gt = num_gt - 1
                    continue
                theta_label = int(theta_longside + 180.5)
                if theta_label == 180:
                    theta_label = 179

                if id > 15 or id < 0:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                    c_x, c_y, longside, shortside, theta_longside))
                if theta_label < 0 or theta_label > 179:
                    print('id problems,问题出现在该图片中:%s' % (i, img_fullname))
                    print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                        c_x, c_y, longside, shortside, theta_longside))
                outline = str(id) + ' ' + ' '.join(list(map(str, bbox))) + ' ' + str(theta_label)
                f_out.write(outline + '\n')

        if num_gt == 0:
            os.remove(os.path.join(dstpath, name + '.txt'))
            os.remove(img_fullname)
            os.remove(fullname)
            print('%s 图片对应的txt不存在有效目标,已删除对应图片与txt' % img_fullname)
    print('已完成文件夹内DOTA数据形式到长边表示法的转换')

def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):

    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

def drawLongsideFormatimg(imgpath, txtpath, dstpath, extractclassname, thickness=2):

    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)
    os.makedirs(dstpath)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        objects = util.parse_longsideformat(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + '.png')
        img_savename = os.path.join(dstpath, name + '_.png')
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        img = cv2.imread(img_fullname)
        for i, obj in enumerate(objects):
            # obj = [classid, x_c_normalized, y_c_normalized, longside_normalized, shortside_normalized, float:0-179]
            class_index = obj[0]
            # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
            rect = longsideformat2cvminAreaRect(obj[1], obj[2], obj[3], obj[4], (obj[5]-179.9))
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.float32(cv2.boxPoints(rect))

            poly[:, 0] = poly[:, 0] * img_w
            poly[:, 1] = poly[:, 1] * img_h
            poly = np.int0(poly)

            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(class_index)],
                             thickness=thickness)
        cv2.imwrite(img_savename, img)

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if theta < -90 or theta >= 0:
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)

def delete(imgpath, txtpath):
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        name = os.path.splitext(os.path.basename(fullname))[0]
        img_fullname = os.path.join(imgpath, name + '.png')
        if not os.path.exists(img_fullname):
            os.remove(fullname)

if __name__ == '__main__':

    dota2LongSideFormat(imgpath='DOTA_demo/images',
                        txtpath='DOTA_demo/yolo_labels',
                        dstpath='DOTA_demo/draw_longside_img',
                        extractclassname=util.classnames_v1_5)

    drawLongsideFormatimg(imgpath='DOTA_demo/images',
                          txtpath='DOTA_demo/yolo_labels',
                          dstpath='DOTA_demo/draw_longside_img',
                          extractclassname=util.classnames_v1_5)
