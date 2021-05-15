# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
#videopath : path of the video file
#frames_path: path of the directory to which the frames are saved
#count: to assign the video order to the frane.
def frameExtractor(videopath,frames_path,count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    
    cap = cv2.VideoCapture(videopath)
    
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no= int(video_length/2.5)
    #frame_no = 20
    
    cap.set(1,frame_no)
    ret,frame=cap.read()
    cv2.imwrite(frames_path + "/%#05d.png" % (count+1), frame)

    #crop frame
    img = cv2.imread(frames_path + "/%#05d.png" % (count+1))
    height, width, c = img.shape
    
    y = int(height/2)
    x = int(width/2)

    cropped_image = img[y-600:y+600 , x-600:x+600]
    cropped_image_path = frames_path + "/%#05d.png" % (count+1)
    cv2.imwrite(cropped_image_path,cropped_image)


