# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:19:10 2019

@author: Saurabh
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

w = [1,0,0,0]
wa = [0,1,0,0]
wd = [0,0,1,0]
s = [0,0,0,1]
starting_value = 1

while True:
    file_name = 'training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along',starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!',starting_value)
        training_data = []
        break
    
def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array
     0   1   2   3  
    [W, WA, WD, S] boolean values.
    '''
    output = [0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'S' in keys:
        output = s 
    elif 'W' in keys:
        output = w
        
    return output


def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print('STARTING!!!')
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,1024,808))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (160,90))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])

            #print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
##            cv2.imshow('window',cv2.resize(screen,(640,360)))
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break

            if len(training_data) % 1000 == 0:
                print(len(training_data))
                
                if len(training_data) == 10000:
                    np.save(file_name,training_data)
                    for i in range(25):
                        print('DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    break
                
            keys = key_check()
            if 'T' in keys:
                if paused:
                    paused = False
                    print('unpaused!')
                    time.sleep(1)
                else:
                    print('Pausing!')
                    paused = True
                    time.sleep(1)


main()