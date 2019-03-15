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

pygame.init()

camera_port = 0
camera = cv2.VideoCapture(camera_port)

pretrained_face_landmark = "shape_predictor_68_face_landmarks.dat" #http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pretrained_face_landmark)

inst = pygame.image.load('instruction.jpg')

infoObject = pygame.display.Info()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen = pygame.display.set_mode((0, 0)) #Костыли так как фуллскрине нельзя alt+tab ,но и открыть окно на весь экрано просто нельзя там рамка(
time.sleep(0.1)  

max_w, max_h = pygame.display.get_surface().get_size()

isFullscreen = True
i = 0
quit = False
dir = "data"
last_image = ''

x = random.randint(0, max_w) #infoObject.current_w   
y = random.randint(0, max_h) #infoObject.current_h   

left_eye = [100, 100]
right_eye = [300, 300]

if not os.path.exists(dir):
    os.makedirs(dir)

while quit == False:
    time.sleep(1 / 30)  

    return_value, frame = camera.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    img = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB") 

    screen.fill((255, 255, 255))
    screen.blit(img, (0,0))
    screen.blit(inst, (600,0))

    pygame.draw.circle(screen, (255, 0, 0), (x, y), 8)
    
    rects = detector(frame, 0)
    
    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)
        for (x_, y_) in shape: #36 42 43 48
            pygame.draw.circle(screen, (0, 255, 0), (x_, y_), 2)


        (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:48]]))
        eyes = frame[y_ - 10:y_ + h_ + 10, x_ - 10:x_ + w_ + 10]
        eyes_frame = pygame.image.frombuffer(eyes.tostring(), (eyes.shape[1], eyes.shape[0]) , "RGB") 
            
        screen.blit(eyes_frame, (0  ,frame.shape[0]))
        
    pygame.display.flip()

    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                quit = True
            if event.key == pygame.K_SPACE:        
                last_image = dir + "/{}_x_{}_y_{}.jpg".format(int(time.time()), x, y)
                pygame.image.save(img, last_image) 
                x = random.randint(0, max_w)
                y = random.randint(0, max_h)
            if event.key == pygame.K_z:
                if os.path.exists(last_image):
                    os.remove(last_image)
        if event.type == pygame.QUIT:
            quit = True
           

#cam.stop()
pygame.quit()