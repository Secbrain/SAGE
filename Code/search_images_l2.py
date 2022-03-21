# -*- coding: utf-8 -*-
import os
from lxml import etree
from threading import *
from time import sleep
import json
import xlrd
from xlrd import xldate_as_tuple
import datetime
import numpy as np
import pandas as pd
import time
import csv
import io
from PIL import Image,ImageDraw,ImageFile
import pytesseract
import cv2
import imagehash
import collections
import photohash
import shutil
import sys

# images_dir = '/mnt/traffic/xzy/wuxian/biased_boundary_attack/out_imagenet_bench.p1m1s1'
# result_out_dir = '/mnt/traffic/xzy/wuxian/biased_boundary_attack'
# images_out_dir = '/mnt/traffic/xzy/wuxian/shengcheng'
# max_dif = 2

def main():
    images_dir = sys.argv[1]
    result_out_dir = sys.argv[2]
    # images_out_dir = sys.argv[3]
    # file_dir = '/mnt/traffic/xzy/wuxian/similar_images/similar_images.json'
    succuss_images = [0 for i in range(5000)]
    images_l2 = [float(-1) for i in range(5000)]
    for file_dir in os.listdir(images_dir):
        if not ('.inprog' in file_dir):
            # shutil.rmtree(os.path.join(path_command,file_dir))
            # remove_dir.append(int(file_dir[:-7]))
            # print(file_dir + "已删除！")
            succuss_images[int(file_dir)] = 1
            image_dir = os.path.join(images_dir,file_dir)
            meta_files = []
            for file in os.listdir(image_dir):
                if 'meta' in file:
                    meta_files.append(file)
            meta_files.sort()
            meta_file_path = os.path.join(image_dir, meta_files[-1])
            meta_file = open(meta_file_path); 
            line = meta_file.readline().strip('\n')
            images_l2[int(file_dir)] = float(line[6:])
            # shutil.copy(os.path.join(os.path.join(images_dir, file_dir),"ae-final.png"), os.path.join(images_out_dir, file_dir + ".png"))
    print("选取完毕！")
    map_image = {}
    map_image["image"] = succuss_images
    map_image["l2"] = images_l2
    data_json = json.dumps(map_image);
    fileObject = open(os.path.join(result_out_dir, 'success_file_result.json'), 'w')
    fileObject.write(data_json)
    fileObject.close()
    print("success_file_result.json文件输出至目的！")

if __name__ == "__main__":
    main();