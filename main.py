import os

import numpy as np
import cv2

import json

from model import efficientdet

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

model, prediction_model = efficientdet(phi=phi, num_classes=90, weighted_bifpn=True, score_threshold=0.01)
prediction_model.load_weights('saved_models/efficientdet-d{}.h5'.format(phi), by_name=True)
prediction_model.summary()
input_shape = prediction_model.input.shape

cam_num = 'cam_01'

f = json.load(open('datasets/{}.json'.format(cam_num)))
zone = f['shapes'][0]['points']
zone = np.array(zone, np.int32)

cap = cv2.VideoCapture('datasets/{}.mp4'.format(cam_num))
while cap.isOpened():
    ret, frame = cap.read()

    dst = frame.copy()
    dst = cv2.polylines(dst, [zone], True, (127,127,127), 5)

    np_input = cv2.resize(frame, (input_shape[1], input_shape[2]))

    boxes, scores, labels = prediction_model.predict(np.expand_dims(np_input, axis=0) / 255.)
    for i, bb in enumerate(boxes[0]):
        lab = labels[0][i]
        if lab == 1 or lab == 2 or lab == 3 or lab == 5 or lab == 7:
            bb = np.array(bb, np.int32)
            bb[0] = bb[0] / np_input.shape[1] * dst.shape[1]
            bb[1] = bb[1] / np_input.shape[0] * dst.shape[0]
            bb[2] = bb[2] / np_input.shape[1] * dst.shape[1]
            bb[3] = bb[3] / np_input.shape[0] * dst.shape[0]
            dst = cv2.rectangle(dst, (bb[0], bb[1]), (bb[2], bb[3]), label_color[lab], 3)

    cv2.imshow('', dst)

    if cv2.waitKey(1) == 27:
        break
cap.release()

