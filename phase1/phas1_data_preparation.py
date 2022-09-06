
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import matplotlib.pyplot as plt
import os
# Initialize the camera

BASE_DIR = ''

camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (720, 480)

# Set the number of frames per second
camera.framerate = 32
 

raw_capture = PiRGBArray(camera, size=(720, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
 
 
 
frame_count = 0
dataset = list()
#save is a variable that when becomes True, after that the code start appending pictures into an arrays to save them at the end.
save = False
# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

     
    # Grab the raw NumPy array representing the image
    image = frame.array
     
    # Display the frame using OpenCV
    cv2.imshow("Frame", image)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        print('d is pushed')
        save = True
        
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        print('q is pushed')
        break
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF
    
    if save == True:
        if frame_count % 3 == 0:
            dataset.append(rgb_image)
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
     
    # If the `q` key was pressed, break from the loop
    
    
    frame_count += 1
    
    
print('out of loop')
for i, j in enumerate(dataset):
    image_name = f'pic{i+1}.jpeg'
    image_path = os.path.join(BASE_DIR, image_name)
    plt.imsave(image_path, dataset[i])
    print('image {} has been successfully saved'.format(i))
    