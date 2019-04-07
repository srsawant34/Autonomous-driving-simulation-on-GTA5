# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:10:51 2019

@author: Saurabh
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from models import inception_v3 as googlenet
from getkeys import key_check

import random

WIDTH = 160
HEIGHT = 90
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'google',EPOCHS)
model = googlenet(WIDTH, HEIGHT, 3, LR, output=4, model_name=MODEL_NAME)
model.load(MODEL_NAME)
t_time = 0.09

def straight():
##    if random.randrange(4) == 2:
##        ReleaseKey(W)
##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


    #ReleaseKey(D)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)
    
def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(S)
    ReleaseKey(A)

    
    

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0,40,1024,808))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = cv2.resize(screen, (160,90))

            prediction = model.predict([screen.reshape(160,90,3)])[0]
            print(prediction)
            moves = list(np.around(prediction))

            turn_thresh = .75
            fwd_thresh = 0.70

            #prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1,  1.8,   1.8, 0.5, 0.5, 0.2])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
                
            elif mode_choice == 1:
                forward_left()
                choice_picked = 'forward+left'
                
            elif mode_choice == 2:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 3:
                reverse()
                choice_picked = 'reverse'

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)

main()       
