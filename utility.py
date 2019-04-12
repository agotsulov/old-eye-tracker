import os
import torch
import train
from imutils import face_utils
import dlib
import cv2
import numpy as np


def load_model(filename, device):
    model = None
    model_file = os.path.exists(filename)
    if model_file:
        print("FOUND MODEL")
        model = train.ConvNet(2).to(device)
        model.load_state_dict(torch.load(filename, map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.eval()
    else:
        print("MODEL NOT FOUND")
    return model


class FaceDetector:
    def __init__(self, predictor_filename="shape_predictor_68_face_landmarks.dat"):
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_filename)
        if self.predictor is None:
            print("PREDICTOR NOT FOUND")

    def predict(self, frame):
        eyes = None
        shape = None

        rects = self.detector(frame, 0)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:48]]))
            eyes = frame[y_ - 10:y_ + h_ + 10, x_ - 10:x_ + w_ + 10]

            if eyes is not None:
                eyes = cv2.resize(eyes, (64, 32))

        return eyes, shape
