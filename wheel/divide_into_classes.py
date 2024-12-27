import argparse

import cv2
import pandas as pd

img_side = 600

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax
                  - outMin))

def divide(img_name):
    # name_labels = 'wheel/sets/test/labels/' + img_name + '_template' + '.txt'
    # name_images = 'wheel/sets/test/images/' + img_name + '_template' + '.png'
    print(img_name)
    index_slash = img_name.rindex('/')
    index_dot = img_name.index('.')
    only_img_name = img_name[index_slash + 1:index_dot]
    df = pd.read_csv('wheel/sets/test/labels/'+only_img_name+'.txt', sep=' ', header = None)
    for i in range(8):
        class_set = []
        for j, line in df.iterrows():
            # print(line)
            if line[0] == i:
                x, y, w, h = -1, -1, -1, -1
                x = num_to_range(float(line[1]), 0, 1, 0, img_side)
                y = num_to_range(float(line[2]), 0, 1, 0, img_side)
                w = num_to_range(float(line[3]), 0, 1, 0, img_side)
                h = num_to_range(float(line[4]), 0, 1, 0, img_side)
                class_set.append([x, y, w, h])
        print(class_set)
        if len(class_set) > 0:
            img = cv2.imread(img_name)
            new_img = ''
            for det in class_set:
                x_s = int(det[0] - det[2]/2)
                y_s = int(det[1] - det[3] / 2)
                x_e = int(det[0] + det[2] / 2)
                y_e = int(det[1] + det[3] / 2)
                cv2.rectangle(img, (x_s, y_s), (x_e, y_e), (0,0,255), 2)
            cv2.imshow('win', img)
            key = cv2.waitKey(0)
    # print(df)

# divide('wheel/sets/test/images/222549_5_png_template.png')
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='wheel/sets/test/images/222549_5_png_template.png', help='img file name')
opt = parser.parse_args()
print(opt)
divide(opt.source)