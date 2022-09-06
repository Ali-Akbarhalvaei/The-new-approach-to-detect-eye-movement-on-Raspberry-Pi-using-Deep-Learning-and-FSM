import cv2
import os

IMAGES_PATH = '/Users/ali.akbarhalvaei/Desktop/presentation/dataset/phase2/5k_2points_ds/images'
LABELS_PATH = '/Users/ali.akbarhalvaei/Desktop/presentation/dataset/phase2/5k_2points_ds/here'

def mouse_click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
		

        font = cv2.FONT_HERSHEY_SIMPLEX
		
        
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 0.3, (255, 0, 0), 1)
        
        cv2.imshow('image', img)

        coordination_list.append(str(x) + ',' +str(y))

        lbl_name = f'{name}.txt'
        lbl_path = os.path.join(LABELS_PATH, lbl_name)
        
        with open(lbl_path, 'w') as f:
            for item in coordination_list:
                f.write("%s\n" % item)
	
        
	

i = 0
for root, _, files in os.walk(IMAGES_PATH, topdown=True):


    for name in files:
        i = i + 1
        print('{} is {}th file out of total {} files'.format(name, i, len(files)))

        if 'jpeg' in name:
            coordination_list = list()
            image_path = os.path.join(root, name)
            img = cv2.imread(image_path)
            cv2.imshow('image', img)

        

            cv2.setMouseCallback('image', mouse_click)

            cv2.waitKey(0)

            cv2.destroyAllWindows()
