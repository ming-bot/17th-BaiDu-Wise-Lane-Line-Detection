from copy import copy
import math
import cv2
import os
import numpy as np
import sys
import time

"""
by Chenlin Ming , Shangyi Li
2022.5.8
"""


class Cruise():
    # initialization hyper parameters
    def __init__(self):
        # record the starting points of the left and right lanes of the previous frame respectively
        self.left_x_base = None
        self.right_x_base = None
        # Reference direction obtained from prior knowledge
        self.left_direction = 0
        self.right_direction = 0
        self.window_height = 0
        self.old_distance = 0
        # Color space segmentation hyper parameters
        self.image_param = {"b_thresh": 150, "a_dark": 120, "a_bright": 160}
        # Sliding window method hyper parameter
        self.slip_window_param = {"window_number": 16, "margin": 32, "minpix": 20}

    # Using color space transformation, the Yellow Lane line is segmented to obtain a binary image
    def image_handle(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        a = lab[:, :, 1]
        b = lab[:, :, 2]
        ret, b = cv2.threshold(b, self.image_param["b_thresh"], 255, cv2.THRESH_BINARY)
        ret, a1 = cv2.threshold(a, self.image_param["a_dark"], 255, cv2.THRESH_BINARY_INV)
        ret, a2 = cv2.threshold(a, self.image_param["a_bright"], 255, cv2.THRESH_BINARY)
        b[((b > 0) & ((a1 > 0) | (a2 > 0)))] = 0
        cv2.medianBlur(b, 5)
        dx = cv2.Sobel(b, ddepth=-1, dx=1, dy=0, ksize=3)
        dx = dx[2:-2, 2:-2]
        return b


    def temporary(self):
        ratio = 0.25
        origin_shape = np.array([480, 640])
        resize_shape = np.array(origin_shape * ratio, dtype='int')
        image = cv2.resize(image, (resize_shape[1], resize_shape[0]))
        frame = 10
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l = lab[:,:,0]
        out = l[frame:-frame]
        return out

    def binary(self, image):
        ratio = 0.25
        origin_shape = np.array([480, 640 ])
        resize_shape = np.array(origin_shape * ratio, dtype='int')
        frame = 10
        lane_thresh = {
            "upper_area": 1000,
            "lower_area": 800,
            "trash_area": 50,
            "min_width": 60,
            "min_height": 70,
            "block_size": 79,
            "edge_height": 30,
            "edge_width": 30,
            "fill": 0.4,
            "rate": 2,
            "height": 100,
            "width": 160
        }
        # 大小重定义
        image = cv2.resize(image, (resize_shape[1], resize_shape[0]))
        # 色彩空间划分
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        b = lab[:, :, 2]
        ret, b = cv2.threshold(b, 140, 255, cv2.THRESH_BINARY)
        b = b[frame:-frame]
        # 去除干扰
        contours, hierarch = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            left = x
            right = x + w
            up = y
            if left > 2 and right < lane_thresh["width"] - 2 and up > 2:
                cv2.drawContours(b, [contours[i]], -1, 0, thickness=-1)
        b ^= 255
        best_contour = None
        contours, hierarch = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count_max = 0
        have_contour = False
        for i in range(len(contours)):
            x_, y, w, h = cv2.boundingRect(contours[i])
            down = y + h
            if down == lane_thresh["height"]:
                count = 0
                for pt in contours[i]:
                    if pt[0, 1] > lane_thresh["height"] - 5:
                        count += 1
                if count >= count_max:
                    best_contour = contours[i]
                    count_max = count
                    have_contour = True
        b[True] = 0
        if have_contour:
            cv2.drawContours(b, [best_contour], -1, 255, -100)
        return b


    def image_binary(self, image):
        """
        extract lane image and convert to binary, 0 for background and 1 for lane
        :param image: original color image
        :return: ouput: binary image
        """
        ratio = 0.25
        origin_shape = np.array([480, 640])
        resize_shape = np.array(origin_shape * ratio, dtype='int')
        image = cv2.resize(image, (resize_shape[1], resize_shape[0]))
        frame = 10
        # image = cv2.resize(image, (resize_shape[1], resize_shape[0]))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS_FULL)
        s = hls[:, :, 2]
        s = s[frame: -frame]
        # cv2.imshow('s', s)
        b = lab[:, :, 2]
        # # filter b for yellow
        ret, b = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((4, 4), np.uint8)
        b = cv2.erode(b, kernel, iterations=2)
        out = b
        out = out[frame:-frame]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(out, connectivity=8)
        s_max = 0
        a = 0
        count = np.zeros([num_labels])
        for i in labels[-1]:
            count[i] += 1
        # 最底下一行中占比最大的连通域
        count_index = sorted(range(len(count)), key=lambda k: count[k], reverse=True)
        cout = labels
        for a in count_index:
            cout = (labels[-1] == a)
            Y = s[-1]
            Y[cout == False] = 0
            summary = np.sum(np.reshape(Y, (Y.size,)))
            num = np.sum(np.reshape(cout, (cout.size,)))
            grey = summary / num
            # break
            if grey < 40:
                break
        cout = (labels == a)
        out[cout] = 255
        out[cout == False] = 0
        kernel_dil = np.ones((8, 8), np.uint8)
        cv2.dilate(out, kernel_dil, iterations=8)
        num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(255 - out, connectivity=8)
        for i in range(1, num_labels_2):
            left = stats_2[i, 0]
            right = stats_2[i, 0] + stats_2[i, 2]
            up = stats_2[i, 1]
            if right < 318 and left > 2 and up > 1:
                cout = (labels_2 == i)
                out[cout] = 255
        # cv2.imshow('out',out)
        # cv2.waitKey()
        return out

    # Sliding window method
    def find_lane_pixels(self, binary_warped):
        # Draw the histogram of the left and right parts of the lower 2/5 part of the image respectively
        leftsight_area = int(binary_warped.shape[0] * 3 / 5)
        rightsight_area = int(binary_warped.shape[0] * 3 / 5)
        midpoint = np.int64(binary_warped.shape[1] // 2)
        left_histogram = np.sum(binary_warped[leftsight_area:, :midpoint], axis=0)
        right_histogram = np.sum(binary_warped[rightsight_area:, midpoint:], axis=0)
        # when we loss the bottom sight, we broaden our sight_area
        # while np.max(left_histogram) == 0:
        #     leftsight_area = leftsight_area//2
        #     left_histogram = np.sum(binary_warped[leftsight_area:,:midpoint], axis=0)
        # while np.max(right_histogram) == 0:
        #     rightsight_area = rightsight_area//2
        #     right_histogram = np.sum(binary_warped[rightsight_area:,midpoint:], axis=0)

        # Create an output image to draw on and visualize the result
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram respectively
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(left_histogram)
        rightx_base = np.argmax(right_histogram) + midpoint
        # if the bottom lose right line, set right line forcely
        if rightx_base == midpoint:
            rightx_base = binary_warped.shape[1] - 1

        # Set height of windows - based on nwindows above and image shape
        self.window_height = np.int64(binary_warped.shape[0] // self.slip_window_param["window_number"])
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_node_list = []
        right_node_list = []

        # Step through the windows one by one
        for window in range(self.slip_window_param["window_number"]):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * self.window_height
            win_y_high = binary_warped.shape[0] - window * self.window_height
            win_xleft_low = leftx_current - self.slip_window_param["margin"]
            win_xleft_high = leftx_current + self.slip_window_param["margin"]
            win_xright_low = rightx_current - self.slip_window_param["margin"]
            win_xright_high = rightx_current + self.slip_window_param["margin"]

            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            # (win_xleft_high,win_y_high),(0,255,0), 2)
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),
            # (win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # If your nonezero pixels found > minpix pixels, recenter next window on their mean position + gradient deviation
            if len(good_left_inds) > self.slip_window_param["minpix"]:
                # leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
                mean_leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
                # Calculated gradient
                good_left_up_inds = ((nonzeroy == np.min(nonzeroy[good_left_inds])) & (nonzerox >= win_xleft_low) &
                                     (nonzerox < win_xleft_high)).nonzero()[0]
                good_left_down_inds = ((nonzeroy == np.max(nonzeroy[good_left_inds])) & (nonzerox >= win_xleft_low) &
                                       (nonzerox < win_xleft_high)).nonzero()[0]
                if np.max(nonzeroy[good_left_inds]) - np.min(nonzeroy[good_left_inds]) > 0:
                    left_node_direction_x = np.int64(
                        (np.mean(nonzerox[good_left_up_inds]) - np.mean(nonzerox[good_left_down_inds])) *
                        (self.window_height) / (np.max(nonzeroy[good_left_inds]) - np.min(nonzeroy[good_left_inds])))
                else:
                    left_node_direction_x = 0
                # Save feature points and corresponding gradients temporarily
                left_x_temporarily = [copy(leftx_current), int((win_y_low + win_y_high) / 2),
                                      copy(left_node_direction_x), copy(self.window_height)]
                leftx_current, gradient = self.Predict_direction(True, leftx_current, mean_leftx_current,
                                                                 left_node_direction_x, window)
                left_x_temporarily[2:] = gradient
            else:
                # No lane line detected
                left_x_temporarily = [copy(leftx_current), int((win_y_low + win_y_high) / 2), 0, 0]
                leftx_current = max(0, leftx_current + int(self.left_direction))
                self.left_direction = self.left_direction * (window / (window + 1))

            if len(good_right_inds) > self.slip_window_param["minpix"]:
                # rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
                mean_rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
                good_right_up_inds = ((nonzeroy == np.min(nonzeroy[good_right_inds])) & (nonzerox >= win_xright_low) &
                                      (nonzerox < win_xright_high)).nonzero()[0]
                good_right_down_inds = ((nonzeroy == np.max(nonzeroy[good_right_inds])) & (nonzerox >= win_xright_low) &
                                        (nonzerox < win_xright_high)).nonzero()[0]
                if np.max(nonzeroy[good_right_inds]) - np.min(nonzeroy[good_right_inds]) > 0:
                    right_node_direction_x = np.int64(
                        (np.mean(nonzerox[good_right_up_inds]) - np.mean(nonzerox[good_right_down_inds])) *
                        (self.window_height) / (np.max(nonzeroy[good_right_inds]) - np.min(nonzeroy[good_right_inds])))
                else:
                    right_node_direction_x = 0
                # Save feature points and corresponding gradients temporarily
                right_x_temporarily = [copy(rightx_current), int((win_y_low + win_y_high) / 2),
                                       copy(right_node_direction_x), copy(self.window_height)]
                rightx_current, gradient = self.Predict_direction(False, rightx_current, mean_rightx_current,
                                                                  right_node_direction_x, window)
                right_x_temporarily[2:] = gradient
            else:
                # No lane line detected
                right_x_temporarily = [copy(rightx_current), int((win_y_low + win_y_high) / 2), 0, 0]
                rightx_current = min(binary_warped.shape[1] - 1, rightx_current + int(self.right_direction))
                self.right_direction = self.right_direction * (window / (window + 1))

            # Judge whether it coincides
            if abs(left_x_temporarily[0] - right_x_temporarily[0]) < 2 * self.slip_window_param["margin"]:
                if len(left_node_list) > 0 and left_node_list[-1][3] == 0:
                    left_node_list.append([0, int((win_y_low + win_y_high) / 2), 0, 0])
                    right_node_list.append(copy(right_x_temporarily))
                elif len(right_node_list) > 0 and right_node_list[-1][3] == 0:
                    right_node_list.append([binary_warped.shape[1] - 1, int((win_y_low + win_y_high) / 2), 0, 0])
                    left_node_list.append(copy(left_x_temporarily))
                else:
                    left_node_list.append([0, int((win_y_low + win_y_high) / 2), 0, 0])
                    right_node_list.append([binary_warped.shape[1] - 1, int((win_y_low + win_y_high) / 2), 0, 0])
            else:
                left_node_list.append(copy(left_x_temporarily))
                right_node_list.append(copy(right_x_temporarily))

        return left_node_list, right_node_list

    def Predict_direction(self, isleft: bool, current: int, mean_current: int, x, window: int):
        if isleft is True:  # is the left lane
            # Unitization
            left_gradient_len = np.sqrt(x ** 2 + self.window_height ** 2)
            length = np.sqrt(self.left_direction ** 2 + self.window_height ** 2)
            Cos = round(x / left_gradient_len, 3) * round(self.left_direction / length, 3) + round(
                self.window_height / left_gradient_len, 3) * round(self.window_height / length, 3)
            if Cos < 0.5:  # cos60°=0.5
                return current + int(self.left_direction), [0, 0]
            else:
                self.left_direction = self.left_direction * (window / (window + 1)) + x * 1 / (window + 1)
                return mean_current + x, [x, self.window_height]
        else:  # is the right lane
            # Unitization
            right_gradient_len = np.sqrt(x ** 2 + self.window_height ** 2)
            length = np.sqrt(self.right_direction ** 2 + self.window_height ** 2)
            Cos = round(x / right_gradient_len, 3) * round(self.right_direction / length, 3) + round(
                self.window_height / right_gradient_len, 3) * round(self.window_height / length, 3)
            if Cos < 0.5:  # cos60°=0.5
                return current + int(self.right_direction), [0, 0]
            else:
                self.right_direction = self.right_direction * (window / (window + 1)) + x * 1 / (window + 1)
                return mean_current + x, [x, self.window_height]

    def Debug_Draw(self, out_img, left_node_list, right_node_list):
        # print("left_node_list len:{}".format(len(left_node_list)))
        # print(left_node_list)
        # print("right_node_list len:{}".format(len(right_node_list)))
        # print(right_node_list)
        for left_node in left_node_list:
            # Graphical display
            if left_node[3] > 0:
                cv2.circle(out_img, center=(left_node[0], left_node[1]), radius=3, color=(255, 255, 0), thickness=6)
                cv2.line(out_img, (left_node[0], left_node[1]),
                         (left_node[0] + left_node[2], left_node[1] - left_node[3]),
                         color=(0, 255, 255), thickness=2)
        for right_node in right_node_list:
            # Graphical display
            if right_node[3] > 0:
                cv2.circle(out_img, center=(right_node[0], right_node[1]), radius=3, color=(255, 255, 0), thickness=6)
                cv2.line(out_img, (right_node[0], right_node[1]),
                         (right_node[0] + right_node[2], right_node[1] - right_node[3]),
                         color=(255, 0, 0), thickness=2)
        cv2.imshow('out', out_img)
        cv2.waitKey(1)

    def handle_later(self, left_node_list, right_node_list,turn_flag):
        left_lane_set = [[]]
        right_lane_set = [[]]
        left_lane_index = 0
        right_lane_index = 0
        iscontinuity = False
        for left_node in left_node_list:
            if left_node[3] == 0:
                if iscontinuity is True:
                    left_lane_index = left_lane_index + 1
                    left_lane_set.append([])
                    iscontinuity = False
            else:
                left_lane_set[left_lane_index].append(left_node)
                iscontinuity = True

        iscontinuity = False
        for right_node in right_node_list:
            if right_node[3] == 0:
                if iscontinuity is True:
                    right_lane_index = right_lane_index + 1
                    right_lane_set.append([])
                    iscontinuity = False
            else:
                right_lane_set[right_lane_index].append(right_node)
                iscontinuity = True

        # drop lane length is less than 3
        total_len = len(left_lane_set)
        for i in range(total_len):
            if len(left_lane_set[total_len - i - 1]) <= 3:
                left_lane_set.pop(total_len - i - 1)
        total_len = len(right_lane_set)
        for i in range(total_len):
            if len(right_lane_set[total_len - i - 1]) <= 3:
                right_lane_set.pop(total_len - i - 1)

        handled_left_lane = []
        handled_right_lane = []
        for left_node_branch in left_lane_set:
            for left_node in left_node_branch:
                left_node.append(len(left_node_branch))
                # if left_node[0] < 0:
                #     left_node[0] = 0
                # elif left_node[0] > 340:
                #     left_node[0] = 340
                handled_left_lane.append(left_node)
        for right_node_branch in right_lane_set:
            for right_node in right_node_branch:
                right_node.append(len(right_node_branch))
                # if right_node[0] > 340:
                #     right_node[0] = 340
                # elif right_node[0] < 0:
                #     right_node[0] = 0
                handled_right_lane.append(right_node)

        total = max(len(handled_left_lane), len(handled_right_lane))
        if total > 0:
            theta = 0  # calculate the angle of longer line
            if len(handled_left_lane) > len(handled_right_lane):
                for left_node_branch in left_lane_set:
                    l = len(left_node_branch)
                    for left_node in left_node_branch:
                        theta += l * left_node[2] * abs(left_node[2]) / (left_node[3]) ** 2 * math.tan(
                            left_node[1] / 240)
            else:
                for right_node_branch in right_lane_set:
                    l = len(right_node_branch)
                    for right_node in right_node_branch:
                        theta += l * right_node[2] * abs(right_node[2]) / (right_node[3]) ** 2 * math.tan(
                            right_node[1] / 240)
            theta = theta / total
            angle = math.atan(theta)
        else:
            angle = 0
        if len(handled_right_lane)>0 and turn_flag:
            if handled_right_lane[-1][2] < -50:
                angle = -5
                print("================================\n"*5)
        return handled_left_lane, handled_right_lane, angle


    def calculate_middle(self, left_node_list, right_node_list):
        halflane_width = 134
        left_to_middle_list = []
        right_to_middle_list = []

        if len(left_node_list) > 0:
            for left_node in left_node_list:
                # (x,y) | (y, -x)
                x1_direction = left_node[3]
                y1_direction = -left_node[2]
                x_direction = x1_direction / math.sqrt(x1_direction ** 2 + y1_direction ** 2)
                y_direction = -y1_direction / math.sqrt(x1_direction ** 2 + y1_direction ** 2)  # - because the cv2 axis
                left_to_middle_list.append(
                    [left_node[0] + halflane_width * x_direction, left_node[1] + halflane_width * y_direction])

        if len(right_node_list) > 0:
            for right_node in right_node_list:
                # (x,y) | (-y, x)
                x2_direction = -right_node[3]
                y2_direction = right_node[2]
                x_direction = x2_direction / math.sqrt(x2_direction ** 2 + y2_direction ** 2)
                y_direction = -y2_direction / math.sqrt(x2_direction ** 2 + y2_direction ** 2)  # - because the cv2 axis
                right_to_middle_list.append(
                    [right_node[0] + halflane_width * x_direction, right_node[1] + halflane_width * y_direction])

        middle_line = left_to_middle_list + right_to_middle_list
        # return left_to_middle_list, right_to_middle_list
        middle_line = sorted(middle_line, key=lambda x: x[1], reverse=True)
        handled_middle_line = []
        jump = 20
        if len(middle_line) > 0:
            count_index = 0
            avg = [0, 0]
            down_y = middle_line[0][1]
            for item in middle_line:
                if item[1] >= down_y - jump:
                    avg[0] += item[0]
                    avg[1] += item[1]
                    count_index += 1
                else:
                    avg[0] = avg[0] / count_index
                    avg[1] = avg[1] / count_index
                    handled_middle_line.append(copy(avg))
                    avg[0] = item[0]
                    avg[1] = item[1]
                    down_y = down_y - jump
                    count_index = 1
        # print(handled_middle_line)
        # print(len(handled_middle_line))
        return handled_middle_line
        

    def velocity_map(self,origin_v):
        valid_min = 13
        valid_max = 25
        real_min = 0
        real_max = 30
        get_v = origin_v*(valid_max-valid_min)/(real_max-real_min) + valid_min*origin_v/abs(origin_v)
        return get_v

    def main(self,image,turn_flag=False):
        tl = [6, 4]
        tr = [138, 9]
        bl = [2, 119]
        br = [157, 117]
        rect = np.array([tl, tr, bl, br], dtype="float32")
        # 变换后对应坐标位置
        dst = np.array([
            [0, 0],
            [48, 0],
            [10, 35.5],
            [41, 36]], dtype="float32") * 340 / 48
        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        # set_speed = 18
        # k1 = 0.5
        # k2 = -0.7
        # k0 = 1
        binary = self.binary(image)
        canny_picture = cv2.Canny(binary, 100, 200)
        # cv2.imshow('binary', binary)
        canny_picture = cv2.warpPerspective(canny_picture, M, (340, 253))
        binary = cv2.warpPerspective(binary, M, (340, 253))
        out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # binary = self.image_handle(small_size)
        left_node_list, right_node_list = self.find_lane_pixels(canny_picture)
        # print('left_lane_len:{}, right_lane_len:{}'.format(len(left_lane), len(right_lane)))
        handled_left_lane, handled_right_lane, angle = self.handle_later(left_node_list, right_node_list, turn_flag)
        # for item in handled_right_lane:
        #     print([item[0], item[1], item[2], item[3]])
        # 这一步是判断是否将单车道线属性检验错误
        if len(handled_right_lane) == 0 and len(handled_left_lane) > 0:
            if 0 < handled_left_lane[1][0] + 10 < 340:
                if binary[handled_left_lane[1][1],handled_left_lane[1][0] + 10] < 50:
                    handled_right_lane = handled_left_lane
                    handled_left_lane = []
        elif len(handled_right_lane) > 0 and len(handled_left_lane) == 0:
            if 0 < handled_right_lane[1][0] - 10 < 340:
                if binary[handled_right_lane[1][1],handled_right_lane[1][0] - 10] < 50:
                    handled_left_lane = handled_right_lane
                    handled_right_lane = []
        middle_lane = self.calculate_middle(handled_left_lane, handled_right_lane)
        # middle_left,middle_right =  self.calculate_middle(handled_left_lane,handled_right_lane)
        # self.Debug_Draw(out, middle_left, middle_right)

        for item in middle_lane:
            cv2.circle(out, center=(int(item[0]), int(item[1])), radius=1, color=(255, 255, 0),
                       thickness=6)
        # for item in middle_left:
        #     cv2.circle(out, center=(int(item[0]), int(item[1])), radius=1, color=(255, 255, 0),
        #                thickness=6)
        # for item in middle_right:
        #     cv2.circle(out, center=(int(item[0]), int(item[1])), radius=1, color=(0, 0, 255),
        #                thickness=6)
        # lab = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
        # b = lab[:,:,2]
        # ret, b = cv2.threshold(b, 140, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow('last', out)
        # cv2.waitKey(1)
        distance = 0
        if len(middle_lane) > 3:
            for j in range(3):
                distance += ((170 - middle_lane[j][0]) * math.tan(middle_lane[j][1] / 240) / 320 * 4) / 3
        elif len(middle_lane) > 0:
            distance += ((170 - middle_lane[0][0]) * math.tan(middle_lane[0][1] / 240) / 320 * 4)
        else:
            pass
        # if abs(angle) > 1 or abs(distance)>0.5:
        #     set_speed = 15
        #     k0 = 0.7
        # else:
        #     set_speed = 15
        #     k0 = 1
        # left_front_speed = set_speed * (k0 + k1 * angle - k2 * distance)
        # left_rear_speed = set_speed * (k0 + k1 * angle + k2 * distance)
        # right_front_speed = set_speed * (k0 - k1 * angle + k2 * distance)
        # right_rear_speed = set_speed * (k0 - k1 * angle - k2 * distance)
        print('angle',angle,'distance',distance)
        return angle,distance
        if self.old_distance == 0 and 0.5 < abs(distance) < 1.5: # illustrate the last picture is cross_road, but have a crash
            distance = distance / 2
            self.old_distance = distance
        if self.old_distance == 0 and abs(distance) > 1.5:
            distance = 0
            self.old_distance = distance
        if distance == 0 and self.old_distance != 0:
            distance = self.old_distance / 3
            self.old_distance = 0
        # if abs(distance - self.old_distance) > 1: # while driving, have a crash
        #     distance = (distance + self.old_distance) / 2
        #     self.old_distance = distance
        if distance > 3:
            distance = 2.5
        elif distance < -3:
            distance = -2.5

        return [left_front_speed, right_front_speed, left_rear_speed, right_rear_speed]

    def move_middle(self,image):
        tl = [6, 4]
        tr = [138, 9]
        bl = [2, 119]
        br = [157, 117]
        rect = np.array([tl, tr, bl, br], dtype="float32")
        # 变换后对应坐标位置
        dst = np.array([
            [0, 0],
            [48, 0],
            [10, 35.5],
            [41, 36]], dtype="float32") * 340 / 48
        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        binary = self.binary(image)
        canny_picture = cv2.Canny(binary, 100, 200)
        binary = cv2.warpPerspective(binary, M, (340, 253))
        canny_picture = cv2.warpPerspective(canny_picture, M, (340, 253))
        binary = binary[-200:]
        canny_picture = canny_picture[-200:]
        left_node_list, right_node_list = self.find_lane_pixels(canny_picture)
        out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        handled_left_lane, handled_right_lane, angle = self.handle_later(left_node_list, right_node_list,False)
        # if len(handled_right_lane) == 0 and len(handled_left_lane) > 0:
        #     if binary[handled_left_lane[1][1],handled_left_lane[1][0] + 10] < 50:
        #         handled_right_lane = handled_left_lane
        #         handled_left_lane = []
        # elif len(handled_right_lane) > 0 and len(handled_left_lane) == 0:
        #     if binary[handled_right_lane[1][1],handled_right_lane[1][0] - 10] < 50:
        #         handled_left_lane = handled_right_lane
        #         handled_right_lane = []
        middle_lane = self.calculate_middle(handled_left_lane, handled_right_lane)
        distance = 0
        if len(middle_lane) > 3:
            for j in range(3):
                distance += ((180 - middle_lane[0][0]) * math.tan(middle_lane[0][1] / 240) / 320 * 4) / 3
        elif len(middle_lane) > 0:
            distance += ((180 - middle_lane[0][0]) * math.tan(middle_lane[0][1] / 240) / 320 * 4)
        else:
            pass
        for item in handled_left_lane:
            cv2.circle(out, center=(int(item[0]), int(item[1])), radius=1, color=(255, 255, 0),
                       thickness=6)
        for item in handled_right_lane:
            cv2.circle(out, center=(int(item[0]), int(item[1])), radius=1, color=(0, 0, 255),
                       thickness=6)
        for item in middle_lane:
            cv2.circle(out, center=(int(item[0]), int(item[1])), radius=1, color=(255, 255, 0),
                       thickness=6)
        # cv2.imshow('task',out)
        # cv2.waitKey(1)
        # stable parameters：k1 = -13 set_speed = 15
        # try parameters: k1 = -11 set_speed = 12
        # this stage aims to smooth the distance change trend


        
        k1 = -13
        k4 = 5
        k5 = 3
        set_speed = 15
        w = k1 * distance
        v_y = set_speed - k4 * abs(angle) - k5 * abs(distance)
        left_front_speed = self.velocity_map(v_y + w)
        left_rear_speed = self.velocity_map(v_y + w)
        right_front_speed = self.velocity_map(v_y - w)
        right_rear_speed = self.velocity_map(v_y - w)
        print("w", w, "v_y", v_y)
        print("angle",angle,"distance",distance)
        # print(left_front_speed, right_front_speed, left_rear_speed, right_rear_speed)
        if abs(angle) < 0.1 and abs(distance) < 0.1:
            flag = True
        else:
            flag = False
        return [left_front_speed, right_front_speed, left_rear_speed, right_rear_speed], flag


if __name__ == '__main__':
    x = PreImgHandle()
    x.main()
