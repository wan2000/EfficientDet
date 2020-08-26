import os

import numpy as np
import cv2

import json

from model import efficientdet

from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes

label = {
    1 : 'bicycle',
    2 : 'car',
    3 : 'motocycle',
    5 : 'bus',
    7 : 'truck'
}

label_color = {
    1 : (0,0,0),
    2 : (255,0,0),
    3 : (0,255,0),
    5 : (0,0,255),
    7 : (255,0,255)
}

phi = 4
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
focus_classes = (1, 2, 3, 5, 7)
image_size = image_sizes[phi]

model, prediction_model = efficientdet(phi=phi, num_classes=90, weighted_bifpn=True, score_threshold=0.01)
prediction_model.load_weights('saved_models/efficientdet-d{}.h5'.format(phi), by_name=True)
prediction_model.summary()
input_shape = prediction_model.input.shape

score_threshold = 0.3

cam_num = 'cam_01'

f = json.load(open('datasets/{}.json'.format(cam_num)))
zone = f['shapes'][0]['points']
zone = np.array(zone, np.int32)

num_classes = 90
classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]

cap = cv2.VideoCapture('datasets/{}.mp4'.format(cam_num))
while cap.isOpened():
    ret, frame = cap.read()

    dst = frame.copy()
    dst = cv2.polylines(dst, [zone], True, (127,127,127), 5)

    frame = frame[:, :, ::-1]
    h, w = frame.shape[:2]

    image, scale = preprocess_image(frame, image_size=image_size)

    boxes, scores, labels = prediction_model.predict([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    # select indices which have a score above the threshold
    indices = np.where(scores[:] > score_threshold)[0]
    

    # select those detections
    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    indices = np.array([], np.int64)

    for i in focus_classes:
        indices = np.concatenate([indices,np.where(labels == i)[0]])

    boxes = boxes[indices]
    labels = labels[indices]
    scores = scores[indices]

    draw_boxes(dst, boxes, scores, labels, colors, classes)

    cv2.imshow('', dst)

    if cv2.waitKey(1) == 27:
        break
cap.release()

