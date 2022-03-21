# -*- coding: utf-8 -*-
import os
# from lxml import etree
from threading import *
from time import sleep
import json
# import xlrd
# from xlrd import xldate_as_tuple
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

# origin_images_dir = '/mnt/traffic/xzy/wuxian/biased_boundary_attack/out_imagenet_bench.p1m1s1'
origin_images_dir = '/mnt/traffic/xzy/wuxian/biased_boundary_attack/out_imagenet_bench'

def main():
    path_command = sys.argv[1]
    remove_dir = []
    for file_dir in os.listdir(path_command):
        if '.inprog' in file_dir:
            shutil.rmtree(os.path.join(path_command,file_dir))
            remove_dir.append(int(file_dir[:-7]))
            print(file_dir + "已删除！")
    print("删除完毕！")
    remove_dir.sort()
    with open("remove_file.txt","w") as f:
        f.write(" ".join([str(x) for x in remove_dir]))

if __name__ == "__main__":
    main();