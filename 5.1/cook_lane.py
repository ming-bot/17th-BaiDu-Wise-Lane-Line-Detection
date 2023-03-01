from copy import copy
import cv2
import os
import numpy as np

"""
by Chenlin Ming , Shangyi Li
2022.5.1
"""
class PreImgHandle():
    # initialization hyper parameters
    def __init__(self):
        # record the starting points of the left and right lanes of the previous frame respectively
        self.left_x_base = None
        self.right_x_base = None
        # Reference direction obtained from prior knowledge
        self.left_direction = 0
        self.right_direction = 0
        # Tolerant jump
        self.tolerance = 50
        # Color space segmentation hyper parameters
        self.image_param = {"b_thresh":150, "a_dark":120, "a_bright":160}
        # Sliding window method hyper parameter
        self.slip_window_param = {"window_number":16, "margin":32, "minpix":50}
    
    # Using color space transformation, the Yellow Lane line is segmented to obtain a binary image
    def image_handle(self, image):
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        a = lab[:,:,1]
        b = lab[:,:,2]
        ret, b = cv2.threshold(b, self.image_param["b_thresh"],255,cv2.THRESH_BINARY)
        ret, a1 = cv2.threshold(a, self.image_param["a_dark"], 255, cv2.THRESH_BINARY_INV)
        ret, a2 = cv2.threshold(a, self.image_param["a_bright"],255,cv2.THRESH_BINARY)
        b[((b > 0) & ((a1 > 0) | (a2 > 0)))] = 0
        cv2.medianBlur(b,5)
        return b

    # Sliding window method
    def find_lane_pixels(self, binary_warped):
        # Draw the histogram of the left and right parts of the lower 2/5 part of the image respectively
        leftsight_area = int(binary_warped.shape[0] * 3/5)
        rightsight_area = int(binary_warped.shape[0] * 3/5)
        midpoint = np.int64(binary_warped.shape[1]//2)
        left_histogram = np.sum(binary_warped[leftsight_area:,:midpoint], axis=0)
        right_histogram = np.sum(binary_warped[rightsight_area:,midpoint:], axis=0)
        # when we loss the bottom sight, we broaden our sight_area
        while np.max(left_histogram) == 0:
            leftsight_area = leftsight_area//2
            left_histogram = np.sum(binary_warped[leftsight_area:,:midpoint], axis=0)
        while np.max(right_histogram) == 0:
            rightsight_area = rightsight_area//2
            right_histogram = np.sum(binary_warped[rightsight_area:,midpoint:], axis=0)
        
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram respectively
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(left_histogram)
        rightx_base = np.argmax(right_histogram) + midpoint
        # if the bottom lose right line, set right line forcely
        if rightx_base == midpoint:
            rightx_base = binary_warped.shape[1] - 1
        
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int64(binary_warped.shape[0]//self.slip_window_param["window_number"])
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        # When the jump is greater than the threshold, it remains stationary
        # leftx_current = leftx_base
        # rightx_current = rightx_base
        if self.left_x_base is None and self.right_x_base is None:
            leftx_current = leftx_base
            self.left_x_base = leftx_base
            rightx_current = rightx_base
            self.right_x_base = rightx_base
        else:
            if abs(leftx_base - self.left_x_base) < self.tolerance:
                leftx_current = leftx_base
                self.left_x_base = leftx_base
            else: leftx_current = self.left_x_base
            if abs(rightx_base - self.right_x_base) < self.tolerance:
                rightx_current = rightx_base
                self.right_x_base = rightx_base
            else: rightx_current = self.right_x_base
        
        # Create empty lists to receive left and right lane pixel indices
        left_node_list = []
        right_node_list = []

        # Step through the windows one by one
        for window in range(self.slip_window_param["window_number"]):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.slip_window_param["margin"]
            win_xleft_high = leftx_current + self.slip_window_param["margin"]
            win_xright_low = rightx_current - self.slip_window_param["margin"]
            win_xright_high = rightx_current + self.slip_window_param["margin"]

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window 
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # If your nonezero pixels found > minpix pixels, recenter next window on their mean position + gradient deviation
            if len(good_left_inds) > self.slip_window_param["minpix"]:
                leftx_current = np.int64(np.mean(nonzerox[good_left_inds]))
                # Calculated gradient
                good_left_up_inds = ((nonzeroy == np.min(nonzeroy[good_left_inds])) & (nonzerox >= win_xleft_low) & 
                (nonzerox < win_xleft_high)).nonzero()[0]
                good_left_down_inds = ((nonzeroy == np.max(nonzeroy[good_left_inds])) & (nonzerox >= win_xleft_low) & 
                (nonzerox < win_xleft_high)).nonzero()[0]
                left_node_direction_x = np.int64(np.mean(nonzerox[good_left_up_inds])) - np.int64(np.mean(nonzerox[good_left_down_inds]))
                left_gradient_len = np.sqrt(left_node_direction_x**2 + (win_y_high - win_y_low)**2)
                # Save feature points and corresponding gradients
                left_node_list.append([copy(leftx_current), int((win_y_low + win_y_high)/2),
                round(copy(left_node_direction_x)/left_gradient_len, 3), round((win_y_high - win_y_low)/left_gradient_len, 3)])
                # # Graphical display
                cv2.circle(out_img, center = (leftx_current, int((win_y_low + win_y_high)/2)), radius=3, color=(255,255,0), thickness=10)
                cv2.line(out_img, (leftx_current, int((win_y_low + win_y_high)/2)), (leftx_current + left_node_direction_x, int((win_y_low + win_y_high)/2) - window_height),
                color=(255,0,0), thickness=3)
                leftx_current = leftx_current + left_node_direction_x
                self.left_direction = self.left_direction * (window/(window + 1)) + left_node_direction_x * 1/(window + 1)
            else:
                # No lane line detected
                left_node_list.append([copy(leftx_current), int((win_y_low + win_y_high)/2), 0, 0])
                self.left_direction = self.left_direction * (window/(window + 1))
                leftx_current = max(0, leftx_current + int(self.left_direction))

            if len(good_right_inds) > self.slip_window_param["minpix"]:
                rightx_current = np.int64(np.mean(nonzerox[good_right_inds]))
                good_right_up_inds = ((nonzeroy == np.min(nonzeroy[good_right_inds])) & (nonzerox >= win_xright_low) & 
                (nonzerox < win_xright_high)).nonzero()[0]
                good_right_down_inds = ((nonzeroy == np.max(nonzeroy[good_right_inds])) & (nonzerox >= win_xright_low) & 
                (nonzerox < win_xright_high)).nonzero()[0]
                right_node_direction_x = np.int64(np.mean(nonzerox[good_right_up_inds])) - np.int64(np.mean(nonzerox[good_right_down_inds]))
                right_gradient_len = np.sqrt(right_node_direction_x**2 + (win_y_high - win_y_low)**2)
                right_node_list.append([copy(rightx_current), int((win_y_low + win_y_high)/2),
                round(copy(right_node_direction_x)/right_gradient_len, 3), round((win_y_high - win_y_low)/right_gradient_len, 3)])
                # # Graphical display
                cv2.circle(out_img, center = (rightx_current, int((win_y_low + win_y_high)/2)), radius=3, color=(255,255,0), thickness=10)
                cv2.line(out_img, (rightx_current, int((win_y_low + win_y_high)/2)), (rightx_current + right_node_direction_x, int((win_y_low + win_y_high)/2) - window_height),
                color=(0,255,255), thickness=3)
                rightx_current = rightx_current + right_node_direction_x
                self.right_direction = self.right_direction * (window/(window + 1)) + self.right_direction * (window/(window + 1))
            else:
                # No lane line detected
                right_node_list.append([copy(rightx_current), int((win_y_low + win_y_high)/2), 0, 0])
                self.right_direction = self.right_direction * (window/(window + 1))
                rightx_current = min(binary_warped.shape[1] - 1, rightx_current + int(self.right_direction))
        
        # print("left_node_list len:{}".format(len(left_node_list)))
        # print(left_node_list)
        # print("right_node_list len:{}".format(len(right_node_list)))
        # print(right_node_list)

        cv2.imshow('out', out_img)
        cv2.waitKey(0)
        return left_node_list, right_node_list, out_img


    def main(self):
        path = os.getcwd()
        for i in range(3,3700):
            image_path = path + '/d1/'+str(i)+'.jpg'
            image = cv2.imread(image_path)
            cropped = image[0:450, 0:680]
            small_size = cv2.resize(cropped, [340, 225])
            binary = self.image_handle(small_size)
            self.find_lane_pixels(binary)

if __name__ == '__main__':
    x = PreImgHandle()
    x.main()