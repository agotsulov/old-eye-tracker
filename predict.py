import pygame
import pygame.camera
import time
import random
import os
import zipfile
from imutils import face_utils
import dlib
import cv2
import numpy as np
import imutils
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import train

pygame.init()

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pretrained_face_landmark = "shape_predictor_68_face_landmarks.dat"
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pretrained_face_landmark)


camera_port = 0
camera = cv2.VideoCapture(camera_port)

infoObject = pygame.display.Info()
print(infoObject)

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

time.sleep(0.1)

max_w, max_h = pygame.display.get_surface().get_size()

isFullscreen = True
quit = False

x_pred = 0
y_pred = 0

model = train.ConvNet(2).to(device)
model.load_state_dict(torch.load('./model.ckpt'))
model.eval()

while not quit:
    time.sleep(1 / 30)

    eyes = None
    shape = None

    return_value, frame = camera.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    img = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")

    screen.fill((255, 255, 255))
    screen.blit(img, (0, 0))

    rects = detector(frame, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        for (x_, y_) in shape:  # 36 42 43 48
            pygame.draw.circle(screen, (0, 255, 0), (x_, y_), 2)

        (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:48]]))
        eyes = frame[y_ - 10:y_ + h_ + 10, x_ - 10:x_ + w_ + 10]

        eyes = cv2.resize(eyes, (64, 32))

        eyes_frame = pygame.image.frombuffer(eyes.tostring(), (eyes.shape[1], eyes.shape[0]), "RGB")

        screen.blit(eyes_frame, (0, frame.shape[0]))

    if eyes is not None:
        eyes = eyes.reshape((1, 3, 64, 32))

        eyes_torch = torch.from_numpy(eyes).float().to(device)
        out = model(eyes_torch)

        out = out.cpu().data.numpy()[0]

        x_pred = out[0]
        y_pred = out[1]

        pygame.draw.circle(screen, (0, 255, 0), (x_pred, y_pred), 12)

    pygame.display.flip()

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit = True
        if event.type == pygame.QUIT:
            quit = True

# cam.stop()
pygame.quit()