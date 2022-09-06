import tensorflow as tf
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


BASE_DIR = ''

#load first model to predict rectangles over eyes to save picture that is inside the rectangles
model = tf.keras.models.load_model('Phase1_model.h5')


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


#when detected becomes true, it means that first model has predicted rectangles
detected = False


camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (720, 480)
# Set the number of frames per second
camera.framerate = 32
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(720, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)


frame_counter = 0
dataset = list()
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    

    image = frame.array
    
    frame = cv2.flip(image, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    if cv2.waitKey(1) & 0xFF == ord('d'):
        
        coords = prediciton(rgb_frame, model)
        detected = True
        
        
        
    if detected == True:

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
        cv2.imshow('concatenated', rgb_eyes)
        
        
        if frame_counter % 3 == 0:
            dataset.append(rgb_eyes)



    frame_counter += 1
    
    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
     
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
     
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        print('q is pushed')
        break


print('out of loop')
for i, j in enumerate(dataset):
    image_name = f'pic{i+1}.jpeg'
    image_path = os.path.join(BASE_DIR, image_name)
    plt.imsave(image_path, dataset[i])
    print('image {} has been successfully saved'.format(i))
    

