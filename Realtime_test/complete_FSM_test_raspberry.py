import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum, auto
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera 
import time

class states(Enum):
    Normal = auto()
    Right = auto()
    Left = auto()
    Up_Ready = auto()
    Normal_Ready = auto()
    Normal_Ready2 = auto()
    Left_Done = auto()
    Right_Done = auto()


class FSM:
    def __init__(self):
        self.__state = states.Normal
        
    def next_to(self, state):
        current_state = self.__state
        
        if current_state == states.Normal_Ready:
            if state == states.Left:
                self.__state = states.Left_Done
            elif state == states.Right:
                self.__state = states.Right_Done
            elif state == states.Up_Ready:
                self.__state = states.Up_Ready
            else:
                self.__state = states.Normal_Ready2
                
        elif current_state == states.Normal_Ready2:
            if state == states.Left:
                self.__state = states.Left_Done
            elif state == states.Right:
                self.__state = states.Right_Done
            elif state == states.Up_Ready:
                self.__state = states.Up_Ready
            else:
                self.__state = states.Normal
        
        elif current_state == states.Up_Ready:
            if state == states.Left:
                self.__state = states.Left_Done
            elif state == states.Right:
                self.__state = states.Right_Done
            elif state == states.Up_Ready:
                pass
            else:
                self.__state = states.Normal_Ready
                
        elif current_state == states.Right_Done:
            if state == states.Up_Ready:
                self.__state = states.Up_Ready
            elif state == states.Right:
                pass
            else:
                self.__state = state
                
                
        elif current_state == states.Left_Done:
            if state == states.Up_Ready:
                self.__state = states.Up_Ready
            elif state == states.Left:
                pass
            else:
                self.__state = state
                
        else:
            self.__state = state
            
                

    def get_state(self):
        return self.__state

first_model = tf.keras.models.load_model('Phase1_model.h5')
second_model = tf.keras.models.load_model('phase2_model.h5')
machine = FSM()

objects = {'normal':states.Normal, 'right':states.Right, 'left':states.Left, 'up':states.Up_Ready}
def prediciton(image, model):

    
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = image_arr / 255
    image_arr = tf.constant(image_arr)
    image = tf.image.resize(image_arr, [224, 224])
    image = tf.reshape(image, [1, 224, 224, 3])

    predicted_points = model(image)
    predicted_points = predicted_points[0]
    
    
    coords_pred = []
    for i in range(0, len(predicted_points) - 1, 2) :
        coords_pred.append([predicted_points[i], predicted_points[i+1]])
        


    coords_pred = np.array(coords_pred, dtype = np.int)

    
    return coords_pred



def predict_iris(image, model):
    
    x = image.shape[0]
    y = image.shape[1]
    
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = image_arr / 255
    image_arr = tf.constant(image_arr)
    image = tf.image.resize(image_arr, [224, 64])
    image = tf.reshape(image, [1, 224, 64, 3])
    
    coords = model(image)
    coordinates = coords[0]
    return np.array(coordinates)



def detect_movement(primary_x, primary_y, cur_x, cur_y):
    direction = ''
    
    diff_X = primary_x - cur_x
    diff_Y = primary_y - cur_y
    
    #print(diff_X, diff_Y)
    direction = detect_direction(diff_X, diff_Y)
    
    return direction




def get_mid_point(x1, x2, y1, y2):
    
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y1) // 2
    
    
    return mid_x, mid_y
    


def detect_direction(diff_x, diff_y):
    direction = ''
        
        
    if diff_y < 10:
        if diff_x < -15:
            direction = 'right'
            machine.next_to(objects[direction])
        
        elif diff_x > 10:
            direction = 'left'
            machine.next_to(objects[direction])
            
    elif diff_y > 15:
        direction = 'up'
        machine.next_to(objects[direction])
        
        
    if direction == '':
        machine.next_to('normal')
            
    return direction


font = cv2.FONT_HERSHEY_SIMPLEX
detected = False
predicted_class = False
First_time = True
dataset = list()
frame_counter = 0
left_close_check = False
direction = ''
classes = ['open', 'left_close', 'right_close']
last_movement = []

camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (720, 480)
# Set the number of frames per second
camera.framerate = 32
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(720, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)


for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

     
    # Grab the raw NumPy array representing the image
    frame = frame.array
    

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        First_time = True
        
        coords = prediciton(rgb_frame, first_model)
        detected = True
        print('d is pushed')
        
        
        
    if detected == True:
        #pred_rec1 = cv2.rectangle(frame, (coords[0,0] - 25, coords[0,1]-70), (coords[1,0] + 20, coords[1,1]+45),(0,0,255),3)
        #pred_rec2 = cv2.rectangle(frame, (coords[2,0] - 25, coords[2,1]-70), (coords[3,0] + 20, coords[3,1]+45),(0,0,255),3)
        left_eye = frame[coords[0,1] - 55 : coords[1,1] + 55, coords[0,0] - 55 : coords[1,0] + 45]
        right_eye = frame[coords[2,1] - 55 : coords[3,1] + 55, coords[2,0] - 55 : coords[3,0] + 45]
        
        
        if left_eye.shape[1] < right_eye.shape[1]:
            left_eye = frame[coords[0,1] - 55 : coords[1,1] + 55, coords[0,0] - 55 : coords[1,0] + 45 + (np.abs(left_eye.shape[1] - right_eye.shape[1]))]
        else:
            right_eye = frame[coords[2,1] - 55 : coords[3,1] + 55, coords[2,0] - 55 : coords[3,0] + 45 + (np.abs(left_eye.shape[1] - right_eye.shape[1]))]
            
            
        if left_eye.shape[0] < right_eye.shape[0]:
            left_eye = frame[coords[0,1] - 55 - (np.abs(left_eye.shape[0] - right_eye.shape[0])) : coords[1,1] + 55, coords[0,0] - 55 : coords[1,0] + 45]
        else:
            right_eye = frame[coords[2,1] - 55 - (np.abs(left_eye.shape[0] - right_eye.shape[0])) : coords[3,1] + 55, coords[2,0] - 55 : coords[3,0] + 45]
    
    
        data = np.concatenate((left_eye, right_eye), axis = 1)
        rgb_eyes = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        
        
        iris_coords = predict_iris(rgb_eyes, second_model)
        iris_coords = list(map(int, iris_coords))


        if First_time == True:
            primary_X, primary_Y = get_mid_point(iris_coords[0], iris_coords[2], iris_coords[1], iris_coords[3])
            First_time = False
        
        else :
            mid_x, mid_y = get_mid_point(iris_coords[0], iris_coords[2], iris_coords[1], iris_coords[3])
            direction = detect_movement(primary_X, primary_Y, mid_x, mid_y)
            
            cv2.putText(rgb_eyes, 
                direction, 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2,
                cv2.LINE_4)
            
            current_state = machine.get_state()
            current_state = f'{current_state}'
            
            cv2.putText(rgb_eyes, 
                current_state, 
                (250, 50), 
                font, 1, 
                (0, 255, 255), 
                2,
                cv2.LINE_4)
            print(current_state)
            
            
        cr1 = cv2.circle(rgb_eyes, (iris_coords[0],iris_coords[1]), radius=1, color=(0, 0, 255), thickness=1)
        cr2 = cv2.circle(rgb_eyes, (iris_coords[2],iris_coords[3]), radius=1, color=(0, 0, 255), thickness=1)
        
        
                
                    
        cv2.imshow('concatenated', rgb_eyes)


        
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





    raw_capture.truncate(0)