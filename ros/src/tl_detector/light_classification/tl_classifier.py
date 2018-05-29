from styx_msgs.msg import TrafficLight
import rospy
from os.path import expanduser
import os
import cv2
import numpy as np
from scipy.ndimage.measurements import label
from math import fabs

def Smoothing(img,size_window=5,color_window=30):
    img_output = np.zeros_like(img)
    cv2.pyrMeanShiftFiltering( img, size_window, color_window, img_output);
    return img_output

def get_color_chennel_mask(channel,low,high):
    channel_mask=np.zeros_like(channel)
    channel_mask[(channel>=low) & (channel<=high)]=1
    return channel_mask

def gen_color_mask(img_rgb,(y_lower,y_upper),(Cb_lower,Cb_upper),(Cr_lower,Cr_upper)):
    YCrCb_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)#.astype(np.float)
    Y_channel = YCrCb_img[:,:,0]
    Cr_channel = YCrCb_img[:,:,1]
    Cb_channel = YCrCb_img[:,:,2]
    #Extract yellow
    #print(Y_channel)
    Y_channel_mask=get_color_chennel_mask(Y_channel,low=y_lower,high=y_upper)
    Cb_channel_mask=get_color_chennel_mask(Cb_channel,low=Cb_lower,high=Cb_upper)
    Cr_channel_mask=get_color_chennel_mask(Cr_channel,low=Cr_lower,high=Cr_upper)
    img_rgb[(Y_channel_mask<1)]=0
    img_rgb[(Cb_channel_mask<1)]=0
    img_rgb[(Cr_channel_mask<1)]=0
    return img_rgb

def open_image(img_rgb):
    kernel = np.ones((3,3),np.uint8)
    output = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
    #output = cv.erode(iimg_rgbmg,kernel,iterations = 1) 
    return output

def get_heat_map(masked_img):
    grey = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
    grey[grey>0]=255
    return grey

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=2,dot_radius=2):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        cv2.circle(imcopy,bbox[2],dot_radius,color,thick)
        #cv2.circle(imcopy,(20,20), 30, (0,0,255), -1)
    # Return the image copy with boxes drawn
    return imcopy

def draw_labeled_bboxes(img, heatmap):
    # Generate the labels from the heat map.
    labels = label(heatmap)
    # Keep a list of bboxes for detected vehicles.
    bboxes = []
    # Iterate through all detected vehicles.
    for tennis in range(1, labels[1]+1):
        # Find pixels with each vehicle label value.
        nonzero = (labels[0] == tennis).nonzero()
        # Identify x and y values of those pixels.
        nonzerox = np.array(nonzero[0])
        nonzeroy = np.array(nonzero[1])
        # Define a bounding box based on the min/max x and y.
        size = (fabs(np.min(nonzeroy)-np.max(nonzeroy)) + fabs(np.min(nonzerox)-np.max(nonzerox)))*0.5
        bbox = ((np.min(nonzeroy), np.min(nonzerox)),
                (np.max(nonzeroy), np.max(nonzerox)),
                (int((np.min(nonzeroy)+np.max(nonzeroy))*0.5),
                 int((np.min(nonzerox)+np.max(nonzerox))*0.5)),size)
        bboxes.append(bbox)
    # Draw the bounding boxes for the detected vehicles.
    img = draw_boxes(img, bboxes)
    # Return the annotated image.
    return img,bboxes

def redlight_detection(img):
    img_copy = img.copy()
    masked_img = gen_color_mask(img_copy,(0,140),(0,110),(175,255))
    morph_img = open_image(masked_img)
    heat_map = get_heat_map(morph_img)
    output,bboxes = draw_labeled_bboxes(img, heat_map)
    size_sum = 0
    average_size=0
    i = 0
    for b in bboxes:
        i=i+1
        size_sum = size_sum + b[3]
    if i > 0:
        average_size = size_sum/i;
    if average_size > 10:
        return True,output
    else:
        return False,output

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        if(0):
            home_dir = expanduser("~")
            self.filepath = home_dir+"/CarND-Capstone/CapturedImages/"
            if not os.path.exists(self.filepath):
                os.makedirs(self.filepath)
        



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        t = rospy.get_time()
        if(0):
            print("Saving")
            filename = self.filepath + "%f.png" % t
            cv2.imwrite(filename, image)
            return TrafficLight.UNKNOWN
        else:
            ifStop, _ = redlight_detection(image)
            StrTime = "%f: "%t
            if(ifStop):
                print(StrTime + "Red Light Detected.")
                return TrafficLight.RED
            else:
                print(StrTime + "No Red Light.")
                return TrafficLight.GREEN
