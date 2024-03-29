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

camera_port = 0
camera = cv2.VideoCapture(camera_port)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pretrained_face_landmark = "shape_predictor_68_face_landmarks.dat"
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pretrained_face_landmark)

inst = pygame.image.load('inst.png')

infoObject = pygame.display.Info()
print(infoObject)
#screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h - 200))
# Костыли так как фуллскрине нельзя alt+tab ,но и открыть окно на весь экрано просто нельзя там рамка(
time.sleep(0.1)  

max_w, max_h = pygame.display.get_surface().get_size()

isFullscreen = False

quit = False
dir = "data"
last_image = ''

x = random.randint(0, max_w)  # infoObject.current_w
y = random.randint(0, max_h)  # infoObject.current_h

model = None

model_file = os.path.exists('./model.pth')
if model_file:
    print("FOUND MODEL")
    model = train.ConvNet(2).to(device)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()
else:
    print("MODEL NOT FOUND")


if not os.path.exists(dir):
    os.makedirs(dir)


while not quit:

    time.sleep(1 / 30)

    eyes = None
    shape = None

    return_value, frame = camera.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    img = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")

    screen.fill((255, 255, 255))
    screen.blit(img, (0, 0))
    screen.blit(inst, (600, 0))

    rects = detector(frame, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        for (x_, y_) in shape:  # 36 42 43 48
            pygame.draw.circle(screen, (0, 255, 0), (x_, y_), 2)
        
        (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:48]]))
        eyes = frame[y_ - 10:y_ + h_ + 10, x_ - 10:x_ + w_ + 10]
        
        if eyes is not None:
            eyes = cv2.resize(eyes, (64, 32))

            eyes_frame = pygame.image.frombuffer(eyes.tostring(), (eyes.shape[1], eyes.shape[0]), "RGB")

            screen.blit(eyes_frame, (0, frame.shape[0]))

    if eyes is not None and model is not None:
        eyes = eyes.reshape((1, 3, 64, 32))
        shape = shape.reshape((1, 68, 2))

        eyes_torch = torch.from_numpy(eyes).float().to(device)
        shape_torch = torch.from_numpy(shape).float().to(device)
        out = model(eyes_torch, shape_torch)

        out = out.cpu().data.numpy()[0]

        x_pred = out[0]
        y_pred = out[1]

        if x_pred < 0:
            x_pred = 0
        if y_pred < 0:
            y_pred = 0
        if x_pred > max_w:
            x_pred = max_w
        if y_pred > max_h:
            y_pred = max_h
            
        pygame.draw.circle(screen, (0, 255, 0), (x_pred, y_pred), 12)

    pygame.draw.circle(screen, (255, 0, 0), (x, y), 8)

    pygame.display.flip()

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit = True
            if event.key == pygame.K_SPACE and eyes is not None:
                last_image = dir + "/{}_x_{}_y_{}.jpg".format(int(time.time()), x, y)
                pygame.image.save(img, last_image)
                x = random.randint(8, max_w - 8)
                y = random.randint(8, max_h - 8)
            if event.key == pygame.K_z:
                if os.path.exists(last_image):
                    os.remove(last_image)
            if event.key == pygame.K_t:
                model = train.train_model()
                model.eval()
            if event.key == pygame.K_f:
                if isFullscreen:
                    screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h))
                    isFullscreen = False
                else:
                    screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN)
                    isFullscreen = True
        if event.type == pygame.QUIT:
            quit = True

# cam.stop()
pygame.quit()
