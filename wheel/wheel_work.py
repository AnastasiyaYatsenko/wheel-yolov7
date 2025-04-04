# import cv2, numpy and matplotlib libraries
import csv
import math
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sympy import false
from wheel.WheelTemplate import *
from wheel.math_functions import *
import os
from SH.SH import PolygonClipper
# from tqdm import tqdm

img_w = 1920
img_h = 1080

win_w = 1920 / 2
win_h = 1080 / 2


class templateApp():
    def __init__(self):
        self.mouseX, self.mouseY = -1, -1
        self.wheel = WheelTemplate()
        with open('wheel/template_params.csv', mode='r') as file:
            # reading the CSV file
            csvFile = csv.reader(file)
            # displaying the contents of the CSV file
            first_row = next(csvFile)  # Compatible with Python 3.x (also 2.7)
            data = list(csvFile)
            self.wheel.set_params(data[0])
        # cv2.namedWindow('image')
        # cv2.setMouseCallback('image', self.sector_click)
        self.template_exist = True
        # print("boop")
        try:
            self.template = cv2.imread('wheel/template.png', cv2.IMREAD_COLOR)
        except:
            print("Template image doesn't exist!")
            self.template_exist = False
        # self.template = cv2.imread('wheel/template.png', cv2.IMREAD_COLOR)
        self.img_path = ""
        self.checkFolders()
        self.is_sector_selected = False
        self.selected_corners = []
        self.selected_center = (-1, -1)
        self.sel_i = -1
        self.marked = []
        # self.automatic_ang_detection()
        self.start = -1
        self.count = -1
        # self.drawWheel()
        # if len(sectors) > 0:
        #     self.automatic_position_sectors(sectors)
        # self.drawWheel()

    def checkFolders(self):
        self.label_folder = 'wheel/sets/analyzed/labels'
        self.image_folder = 'wheel/sets/analyzed/images'
        self.faulty_folder = 'wheel/sets/faulty'
        self.sector_folder = 'wheel/sectors'
        isExistLabels = os.path.exists('wheel/sets/analyzed/labels')
        isExistImages = os.path.exists('wheel/sets/analyzed/images')
        if not os.path.exists(self.label_folder):
            os.makedirs(self.label_folder)
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        if not os.path.exists(self.faulty_folder):
            os.makedirs(self.faulty_folder)
        if not os.path.exists(self.sector_folder):
            os.makedirs(self.sector_folder)
        for i in range(len(self.wheel.wheel_class)):
            class_name = self.wheel.wheel_class_names[i]
            sector_subfolder = f'{self.sector_folder}/{class_name}'
            if not os.path.exists(sector_subfolder):
                os.makedirs(sector_subfolder)


    def set_img(self, img='', sectors=[]):
        if not self.template_exist:
            return -1

        self.img_path = ""
        if img == '':
            self.img_path = "wheel/img/222549_5.png"
        else:
            self.img_path = img
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            print(f"Can't read the image {self.img_path}")
            return -1
        if os.name == 'nt':
            index_slash = self.img_path.rfind('\\')
        else:
            index_slash = self.img_path.rfind('/')
        index_dot = self.img_path.find('.')
        self.img_name = self.img_path[index_slash + 1:index_dot]
        self.is_sector_selected = False
        self.selected_corners = []
        self.selected_center = (-1, -1)
        self.sel_i = -1
        self.marked = []
        # print("2")
        res = self.automatic_ang_detection()
        if res == -1:
            print(f"Template not found on image")
            return -1
        # print("3")
        self.start = -1
        self.count = -1
        self.drawWheel()
        # print("4")
        if len(sectors) > 0:
            self.automatic_position_sectors(sectors)
        # print("5")
        self.drawWheel()
        return 1

    def save_faulty(self):
        name_faulty = self.faulty_folder + '/' + self.img_name + '_faulty' + '.png'
        cv2.imwrite(name_faulty, self.img)  # save to file

    def automatic_ang_detection(self):
        self.wheel.ang_rotate = 0
        h, w = self.template.shape[:2]

        result = cv2.matchTemplate(self.img, self.template, cv2.TM_CCOEFF_NORMED)
        # threshold = 0.39
        threshold = 0.7
        locations = np.where(result >= threshold)

        if len(locations) <= 0:
            # self.save_faulty()
            return -1

        # image_with_lines = image.copy()
        p1 = self.wheel.center[:]

        object_center_x = locations[1][0] + w // 2
        object_center_y = locations[0][0] + h // 2
        self.object_center_x = object_center_x
        self.object_center_y = object_center_y

        ang = angle_between_vectors((object_center_x - p1[0]), (p1[1] - object_center_y), -50, 0)
        self.wheel.ang_rotate = ang

    def drawWheel(self):
        # img = cv2.imread(self.img_path)
        # img = cv2.imread("img/222549_5.png")
        # half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # print("d 1")
        # cv2.circle(self.img, self.wheel.center, self.wheel.R_full, (0, 0, 255), 3)
        # cv2.circle(self.img, self.wheel.center, self.wheel.R_outer, (0, 255, 0), 2)
        # cv2.circle(self.img, self.wheel.center, self.wheel.R_inner, (0, 255, 0), 2)
        # cv2.line(self.img, (self.object_center_x, self.object_center_y), (self.wheel.center[0], self.wheel.center[1]),
        #          (150, 255, 0), 2)
        # Displaying image using plt.imshow() method

        # sectors = self.wheel.getSectors()
        angle = self.wheel.ang_rotate
        # i = 0
        # print("d 2")
        for s in self.wheel.sectors:
            p1 = self.wheel.center[:]
            p2 = get_point_on_circle(p1, self.wheel.R_full, angle)
            # if i == 0:
            #     cv2.line(self.img, p1, p2, (255, 255, 255), 2)
            # else:
            #     cv2.line(self.img, p1, p2, (0, 0, 255), 2)
            # i += 1

            # print(f"d 2.{i} - {s.name}")

            if s.name == 'S' or s.name == 'P' or s.name == 'T':
                pc1 = get_point_on_circle(p1, self.wheel.R_inner, angle)
                pc2 = get_point_on_circle(p1, self.wheel.R_outer, angle)
                p_text = get_point_on_circle(p1, self.wheel.R_outer + 20, (angle + s.ang / 2))
                angle += s.ang
                pc3 = get_point_on_circle(p1, self.wheel.R_outer, angle)
                pc4 = get_point_on_circle(p1, self.wheel.R_inner, angle)
                corners = [pc1, pc2, pc3, pc4]
                s_c = intersection_point(pc1, pc3, pc2, pc4)

                s.set_coords(s_c, corners)
                # s.center = s_c[:]
                # for i in range(len(corners)):
                #     s.corners[i] = corners[i][:]

                # cv2.line(self.img, pc1, pc3, (255, 0, 0), 2)
                # cv2.line(self.img, pc2, pc4, (255, 0, 0), 2)
                #
                # cv2.putText(self.img, s.name, p_text, cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                side_ang = (s.ang - s.sector_ang) / 2
                angle += side_ang
                p3 = get_point_on_circle(p1, self.wheel.R_full, angle)
                # cv2.line(self.img, p1, p3, (0, 255, 0), 2)

                pc1 = get_point_on_circle(p1, self.wheel.R_inner, angle)
                pc2 = get_point_on_circle(p1, self.wheel.R_outer, angle)
                p_text = get_point_on_circle(p1, self.wheel.R_outer + 20, (angle + s.sector_ang / 2))

                angle += s.sector_ang
                p4 = get_point_on_circle(p1, self.wheel.R_full, angle)
                # cv2.line(self.img, p1, p4, (0, 255, 0), 2)

                pc3 = get_point_on_circle(p1, self.wheel.R_outer, angle)
                pc4 = get_point_on_circle(p1, self.wheel.R_inner, angle)
                corners = [pc1, pc2, pc3, pc4]
                s_c = intersection_point(pc1, pc3, pc2, pc4)
                s.set_coords(s_c, corners)

                # cv2.line(self.img, pc1, pc3, (255, 0, 0), 2)
                # cv2.line(self.img, pc2, pc4, (255, 0, 0), 2)
                # cv2.putText(self.img, s.name, p_text, cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (0, 255, 0), 2, cv2.LINE_AA)

                angle += side_ang

        # if self.is_sector_selected:
        #     fill_color = [127, 256, 32]  # any BGR color value to fill with
        #     mask_value = 255  # 1 channel white (can be any non-zero uint8 value)
        #
        #     # contours to fill outside of
        #
        #     contours = [np.array([[self.selected_corners[0][0], self.selected_corners[0][1]],
        #                           [self.selected_corners[1][0], self.selected_corners[1][1]],
        #                           [self.selected_corners[2][0], self.selected_corners[2][1]],
        #                           [self.selected_corners[3][0], self.selected_corners[3][1]]])]
        #
        #     cv2.line(self.img, self.selected_corners[0], self.selected_corners[1], (0, 0, 255), 5)
        #     cv2.line(self.img, self.selected_corners[1], self.selected_corners[2], (0, 0, 255), 5)
        #     cv2.line(self.img, self.selected_corners[2], self.selected_corners[3], (0, 0, 255), 5)
        #     cv2.line(self.img, self.selected_corners[3], self.selected_corners[0], (0, 0, 255), 5)

        # cv2.imshow('Image Window', img)
        # w = img_w / 2
        # h = img_h / 2
        # size = (w, h)
        # resized_image = cv2.resize(img, (960, 540))
        # cv2.imshow('image', resized_image) # un-comment for image show

    def delete_sel(self):
        self.is_sector_selected = False
        self.selected_corners = []
        self.selected_center = (-1, -1)
        self.sel_i = -1
        self.marked = []

    def automatic_position_sectors(self, det_sectors):
        detected_wheel = [0 for i in range(len(self.wheel.sectors))]

        # find the detected sector with minimal x
        # find the corresponding wheel sector
        # go through all sectors from the found wheel sector

        sorted_det_sectors = sorted(det_sectors, key=lambda x: (x[0][0]))
        first = -1
        point = Point(sorted_det_sectors[0][0][0], sorted_det_sectors[0][0][1])

        for i in range(len(self.wheel.sectors)):
            polygon = Polygon(self.wheel.sectors[i].corners)
            r = polygon.contains(point)
            if r:
                first = i
                break

        actual_count = 0
        for i in range(len(self.wheel.sectors)):
            for d_s in sorted_det_sectors:
                point = Point(d_s[0][0], d_s[0][1])
                polygon = Polygon(self.wheel.sectors[i].corners)
                r = polygon.contains(point)
                if r and detected_wheel[i] == 0:
                    detected_wheel[i] = d_s[1]
                    actual_count += 1

        arr1 = detected_wheel[first:first + actual_count]
        arr2 = detected_wheel[0:(actual_count - len(arr1))]
        det = arr1 + arr2
        for i in range(len(det)):
            if not det[i] in ["1", "2", "5"]:
                det[i] = "?"
        self.count = actual_count
        self.position_sectors(det, first)

    def position_sectors(self, marked, start_i):
        len_marked_ = len(marked) - 1
        joined_wheel = ''.join(self.wheel.wheel_nums + self.wheel.wheel_nums[:len_marked_])
        joined_marked = ''.join(marked)
        joined_wheel = re.sub('10', '9', joined_wheel)
        reg = re.sub('\?', '[1-9,S,T,P]', joined_marked)
        p = re.compile(reg)
        part_list = re.findall(p, joined_wheel)
        num = len(part_list)
        if num > 1 or num <= 0:
            return -1
        i = joined_wheel.find(part_list[0])
        self.start = i
        ang_shift = 0
        min_i = min(i, start_i)
        max_i = max(i, start_i)
        for j in range(min_i, max_i):
            ang_shift += self.wheel.sectors[j].ang
        if i > start_i:
            ang_shift = -ang_shift
        self.wheel.ang_rotate += ang_shift
        self.wheel.ang_rotate = normalize(self.wheel.ang_rotate)
        self.save_params()
        return ang_shift

    def position_sectors_manual(self):
        len_marked_ = len(self.marked) - 1
        joined_wheel = ''.join(self.wheel.wheel_nums + self.wheel.wheel_nums[:len_marked_])
        joined_marked = ''.join(self.marked)
        joined_wheel = re.sub('10', '9', joined_wheel)
        reg = re.sub('\?', '[1-9,S,T,P]', joined_marked)
        p = re.compile(reg)
        part_list = re.findall(p, joined_wheel)
        num = len(part_list)
        if num > 1 or num <= 0:
            return -1
        i = joined_wheel.find(part_list[0])
        ang_shift = 0
        min_i = min(i, self.sel_i)
        max_i = max(i, self.sel_i)
        for j in range(min_i, max_i):
            ang_shift += self.wheel.sectors[j].ang
        if i > self.sel_i:
            ang_shift = -ang_shift
        self.delete_sel()
        self.wheel.ang_rotate += ang_shift
        self.wheel.ang_rotate = normalize(self.wheel.ang_rotate)
        return ang_shift

    def write_to_file(self, segmentation=false):
        # print("in write to file")
        # index_slash = self.img_path.rfind('/')
        # index_dot = self.img_path.find('.')
        # img_name = self.img_path[index_slash+1:index_dot]
        name_labels = self.label_folder + '/' + self.img_name + '_template' + '.txt'
        name_images = self.image_folder + '/' + self.img_name + '_template' + '.png'
        # print(name_labels)
        # orig_img = cv2.imread(self.img_path)
        img = cv2.resize(self.img, (640, 640))
        cv2.imwrite(name_images, img) # save to file
        counts = [0 for i in range(8)]
        with open(name_labels, 'w', newline='') as f:
            for j in range(self.start, self.start + self.count):
                i = j
                if j >= len(self.wheel.sectors):
                    i = j - len(self.wheel.sectors)
                if (not 0 < self.wheel.sectors[i].center[0] < img_w) or (
                        not 0 < self.wheel.sectors[i].center[1] < img_h):
                    continue
                label = self.wheel.wheel_class[self.wheel.sectors[i].name]
                min_x = min(self.wheel.sectors[i].corners[0][0],
                            self.wheel.sectors[i].corners[1][0],
                            self.wheel.sectors[i].corners[2][0],
                            self.wheel.sectors[i].corners[3][0])
                max_x = max(self.wheel.sectors[i].corners[0][0],
                            self.wheel.sectors[i].corners[1][0],
                            self.wheel.sectors[i].corners[2][0],
                            self.wheel.sectors[i].corners[3][0])
                min_y = min(self.wheel.sectors[i].corners[0][1],
                            self.wheel.sectors[i].corners[1][1],
                            self.wheel.sectors[i].corners[2][1],
                            self.wheel.sectors[i].corners[3][1])
                max_y = max(self.wheel.sectors[i].corners[0][1],
                            self.wheel.sectors[i].corners[1][1],
                            self.wheel.sectors[i].corners[2][1],
                            self.wheel.sectors[i].corners[3][1])
                w = max_x - min_x
                h = max_y - min_y
                center_x = num_to_range(self.wheel.sectors[i].center[0], 0, img_w, 0, 1)
                center_y = num_to_range(self.wheel.sectors[i].center[1], 0, img_h, 0, 1)
                width = num_to_range(w, 0, img_w, 0, 1)
                height = num_to_range(h, 0, img_h, 0, 1)
                str = f"{label} {center_x} {center_y} {width} {height}\n"
                flag = True

                # cutting_corners = self.wheel.sectors[i].corners
                cutting_corners = []

                if segmentation:
                    str = f"{label}"
                    # cutting_corners = []

                    x1 = self.wheel.sectors[i].corners[0][0]
                    y1 = self.wheel.sectors[i].corners[0][1]
                    x2 = self.wheel.sectors[i].corners[1][0]
                    y2 = self.wheel.sectors[i].corners[1][1]
                    x3 = self.wheel.sectors[i].corners[2][0]
                    y3 = self.wheel.sectors[i].corners[2][1]
                    x4 = self.wheel.sectors[i].corners[3][0]
                    y4 = self.wheel.sectors[i].corners[3][1]

                    subject_polygon = np.array([[x4, y4], [x3, y3], [x2, y2], [x1, y1]])
                    clipping_polygon = np.array([[0, 0], [0, img_h], [img_w, img_h], [img_w, 0]])

                    clip = PolygonClipper(warn_if_empty=True)
                    clipped_polygon = clip(subject_polygon, clipping_polygon)
                    if len(clipped_polygon) > 0:
                        for coord in clipped_polygon:
                            x = num_to_range(int(coord[0]), 0, img_w, 0, 1)
                            y = num_to_range(int(coord[1]), 0, img_h, 0, 1)
                            cutting_corners.append((int(coord[0]), int(coord[1])))
                            str += f" {x} {y}"
                        # str += "\n"
                    else:
                        for coord in subject_polygon:
                            x = num_to_range(int(coord[0]), 0, img_w, 0, 1)
                            y = num_to_range(int(coord[1]), 0, img_h, 0, 1)
                            cutting_corners.append((int(coord[0]), int(coord[1])))
                            str += f" {x} {y}"
                    str += "\n"
                else:
                    label = self.wheel.wheel_class[self.wheel.sectors[i].name]
                    min_x = min(self.wheel.sectors[i].corners[0][0],
                                self.wheel.sectors[i].corners[1][0],
                                self.wheel.sectors[i].corners[2][0],
                                self.wheel.sectors[i].corners[3][0])
                    max_x = max(self.wheel.sectors[i].corners[0][0],
                                self.wheel.sectors[i].corners[1][0],
                                self.wheel.sectors[i].corners[2][0],
                                self.wheel.sectors[i].corners[3][0])
                    min_y = min(self.wheel.sectors[i].corners[0][1],
                                self.wheel.sectors[i].corners[1][1],
                                self.wheel.sectors[i].corners[2][1],
                                self.wheel.sectors[i].corners[3][1])
                    max_y = max(self.wheel.sectors[i].corners[0][1],
                                self.wheel.sectors[i].corners[1][1],
                                self.wheel.sectors[i].corners[2][1],
                                self.wheel.sectors[i].corners[3][1])
                    w = max_x - min_x
                    h = max_y - min_y
                    center_x = num_to_range(self.wheel.sectors[i].center[0], 0, img_w, 0, 1)
                    center_y = num_to_range(self.wheel.sectors[i].center[1], 0, img_h, 0, 1)
                    width = num_to_range(w, 0, img_w, 0, 1)
                    height = num_to_range(h, 0, img_h, 0, 1)
                    str = f"{label} {center_x} {center_y} {width} {height}\n"

                if flag:
                    f.write(str) # save to file
                else:
                    continue # no need to save this sector

                if segmentation:
                    x_array = []
                    y_array = []
                    for coord in cutting_corners:
                        x_array.append(coord[0])
                        y_array.append(coord[1])

                    # print(cutting_corners)
                    # print(x_array)
                    # print(y_array)
                    top_left_x = min(x_array)
                    top_left_y = min(y_array)
                    bot_right_x = max(x_array)
                    bot_right_y = max(y_array)

                    b_channel, g_channel, r_channel = cv2.split(self.img)
                    alpha_channel = np.ones(b_channel.shape,
                                            dtype=b_channel.dtype) * 255  # creating a dummy alpha channel image.

                    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

                    mask = np.zeros(img_BGRA.shape, dtype=np.uint8)
                    roi_corners = np.array([cutting_corners], dtype=np.int32)
                    # fill the ROI so it doesn't get wiped out when the mask is applied
                    channel_count = img_BGRA.shape[2]  # i.e. 3 or 4 depending on your image
                    ignore_mask_color = (255,) * channel_count
                    cv2.fillConvexPoly(mask, roi_corners, ignore_mask_color)
                    # from Masterfool: use cv2.fillConvexPoly if you know it's convex

                    # apply the mask
                    masked_image = cv2.bitwise_and(img_BGRA, mask)
                    sec = masked_image[top_left_y:bot_right_y + 1, top_left_x:bot_right_x + 1]

                    # cv2.imshow('sec', masked_image)
                    # if 0 <= top_left_x <= img_w and 0 <= top_left_y <= img_h and 0 <= bot_right_x <= img_w and 0 <= bot_right_y <= img_h:
                    class_name = self.wheel.wheel_class_names[label]
                    name = f'{self.sector_folder}/{class_name}/{self.img_name}_{class_name}_{counts[label]:02}.png'
                    counts[label] += 1
                    cv2.imwrite(name, sec) # save to file

    def get_visible_sectors(self):
        # sectors = self.wheel.getSectors()
        print("--- sectors ---")
        for s in self.wheel.sectors:
            if 0 <= s.center[0] <= img_w and 0 <= s.center[1] <= img_h:
                print(f"{s.name}: {s.corners}, {s.center}")
        print("--- end ---")

    def which_sector(self, x, y):
        # x = num_to_range(self.mouseX, 0, win_w, 0, img_w)
        # y = num_to_range(self.mouseY, 0, win_h, 0, img_h)
        for i in range(len(self.wheel.sectors)):
            point = Point(x, y)
            polygon = Polygon(self.wheel.sectors[i].corners)
            r = polygon.contains(point)
            # sel = False
            if r:
                return i
                # self.marked = []
                # if self.is_sector_selected:
                #     sel_polygon = Polygon(self.selected_corners)
                #     sel = sel_polygon.contains(point)
                #     if sel and r:
                #         self.is_sector_selected = False
                #         self.selected_corners = []
                # if r and not sel:
                # self.is_sector_selected = True
                # self.selected_corners = self.wheel.sectors[i].corners[:]
                # return i
        return -1

    def click_in_sector(self):
        x = num_to_range(self.mouseX, 0, win_w, 0, img_w)
        y = num_to_range(self.mouseY, 0, win_h, 0, img_h)
        i = self.which_sector(x, y)
        # self.is_sector_selected = True
        # self.selected_corners = self.wheel.sectors[i].corners[:]
        # return i
        # for i in range(len(self.wheel.sectors)):
        #     point = Point(x, y)
        #     polygon = Polygon(self.wheel.sectors[i].corners)
        #     r = polygon.contains(point)
        #     sel = False
        if i != -1:
            point = Point(x, y)
            polygon = Polygon(self.wheel.sectors[i].corners)
            self.marked = []
            if self.is_sector_selected:
                sel_polygon = Polygon(self.selected_corners)
                sel = polygon.contains(point)
                if sel:
                    self.is_sector_selected = False
                    self.selected_corners = []
                else:
                    self.is_sector_selected = True
                    self.selected_corners = self.wheel.sectors[i].corners[:]
                    return i
        return -1

    def sector_click(self, event, x, y, flags, param):
        # self.mouseX, self.mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            # cv2.circle(img, (x, y), 100, (255, 0, 0), -1)
            self.mouseX, self.mouseY = x, y
            self.sel_i = self.click_in_sector()
            self.drawWheel()
            print(f"click: {self.mouseX}, {self.mouseY} - {self.sel_i}")

    def save_params(self):
        with open('wheel/template_params.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # field names
            fields = ['R_full', 'R_outer', 'R_inner', 'ang_rotate', 'center_x', 'center_y']
            # data rows of csv file
            rows = [[self.wheel.R_full, self.wheel.R_outer, self.wheel.R_inner,
                     self.wheel.ang_rotate, self.wheel.center[0], self.wheel.center[1]]]
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)

    def run(self):
        key = cv2.waitKey(0)  # Wait for a key press to close the window
        while key != ord('q'):
            if not self.is_sector_selected:
                if key == ord('8'):
                    self.wheel.move(0, -1)
                elif key == ord('4'):
                    self.wheel.move(-1, 0)
                elif key == ord('2'):
                    self.wheel.move(0, 1)
                elif key == ord('6'):
                    self.wheel.move(1, 0)
                elif key == ord('-'):
                    self.wheel.scale(-1)
                elif key == ord('+'):
                    self.wheel.scale(1)
                elif key == ord('['):
                    self.wheel.scaleO(-1)
                elif key == ord(']'):
                    self.wheel.scaleO(1)
                elif key == ord(','):
                    self.wheel.scaleI(-1)
                elif key == ord('.'):
                    self.wheel.scaleI(1)
                elif key == ord('z'):
                    self.wheel.rotate(-0.1)
                elif key == ord('c'):
                    self.wheel.rotate(0.1)
                elif key == ord('a'):
                    self.wheel.rotate(-6.67)
                elif key == ord('d'):
                    self.wheel.rotate(6.67)
                elif key == ord('s'):
                    self.save_params()
                    # with open('template_params.csv', 'w', newline='') as csvfile:
                    #     csvwriter = csv.writer(csvfile, delimiter=',',
                    #                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    #     # field names
                    #     fields = ['R_full', 'R_outer', 'R_inner', 'ang_rotate', 'center_x', 'center_y']
                    #     # data rows of csv file
                    #     rows = [[self.wheel.R_full, self.wheel.R_outer, self.wheel.R_inner,
                    #              self.wheel.ang_rotate, self.wheel.center[0], self.wheel.center[1]]]
                    #     csvwriter.writerow(fields)
                    #     csvwriter.writerows(rows)
                elif key == ord('v'):
                    self.get_visible_sectors()
            else:
                if key == ord('1'):
                    self.marked.append('1')
                elif key == ord('2'):
                    self.marked.append('2')
                elif key == ord('5'):
                    self.marked.append('5')
                elif key == ord('8'):
                    self.marked.append('8')
                elif key == ord('0'):
                    if self.marked[-1] == '1':
                        self.marked[-1] = '9'
                        # self.marked[-1] = '10'
                elif key == ord('s'):
                    self.marked.append('S')
                elif key == ord('t'):
                    self.marked.append('T')
                elif key == ord('p'):
                    self.marked.append('P')
                elif key == ord('m'):
                    print(self.marked)
                elif key == ord(' '):
                    self.marked.append('?')
                elif key == ord('v'):
                    self.get_visible_sectors()
                self.position_sectors_manual()
            self.drawWheel()
            key = cv2.waitKey(0)
        cv2.destroyAllWindows()  # Close the window
