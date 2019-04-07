# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:46:09 2019

@author: Saurabh
"""

import numpy as np
import cv2

train_data = np.load('training_data-1.npy')
#print(train_data.shape)

for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break