# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from numpy import genfromtxt
from scipy import spatial

import glob
import cv2
import numpy as np
import os
import tensorflow as tf


def find_gesture_number(vect, penul_layer):
    cost_dist = []

    for p in penul_layer:
        cost_dist.append(spatial.distance.cosine(vect,p))
        ges = cost_dist.index(min(cost_dist))+1
    
    return ges


def getPenultimateLayer(frames_path,file_name):
    files_list = []

    path = os.path.join(frames_path,"*.png")
    frames = glob.glob(path)
    frames.sort()
    files_list = frames

    prediction_vector = get_vectors_for_frames(files_list)
    np.savetxt(file_name, prediction_vector, delimiter=",")



def get_vectors_for_frames(files_list):

    vectors = []
    videos = []

    prediction_model = HandShapeFeatureExtractor.get_instance()

    for frame in files_list:
        img = cv2.imread(frame)
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        results = prediction_model.extract_feature(img)
        results = np.squeeze(results)

        vectors.append(results)
        videos.append(os.path.basename(frame))
    
    return vectors




# =============================================================================
# Get the penultimate layer for training data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

video_folder = os.path.join('traindata')
videos_path = os.path.join(video_folder,"*.mp4")
all_videos = glob.glob(videos_path)

video_files_list = []
video_files_list = all_videos

count = 0

for video in video_files_list:
    frames_path= os.path.join(video_folder,"frames")
    frameExtractor(video, frames_path, count)
    count += 1

pendata_file_name = 'training_vector.csv'
frames_path = os.path.join(video_folder,"frames")

getPenultimateLayer(frames_path, pendata_file_name)

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

video_folder = os.path.join('test')
videos_path = os.path.join(video_folder,"*.mp4")
all_videos = glob.glob(videos_path)

video_files_list = []
video_files_list = all_videos

count = 0

for video in video_files_list:
    frames_path = os.path.join(video_folder,"frames")
    frameExtractor(video, frames_path, count)
    count += 1

pendata_file_name2 = 'test_vector.csv'
frames_path = os.path.join(video_folder,"frames")

getPenultimateLayer(frames_path, pendata_file_name2)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

training_data = genfromtxt(pendata_file_name, delimiter=',')
test_data = genfromtxt(pendata_file_name2, delimiter=',')

res = []

for x in test_data:
    res.append(find_gesture_number(x, training_data))

np.savetxt('Results.csv', res, delimiter=",", fmt='% d')
