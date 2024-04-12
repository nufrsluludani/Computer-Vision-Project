# import libs
import cv2
import numpy as np

# read images
# img = cv2.imread('weed.png')


# read webcam
cap = cv2.VideoCapture(1)

# set up camera settings
cap.set(3, 640)
cap.set(4, 480)

# manage classNames
classNames = []
classFile = 'coco.names'

# put coco data into classNames list
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  # read file, split line by name into class names
    print(classNames)  # check if classes are in the list

# define paths
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# setting up bounding box and ids
net = cv2.dnn.DetectionModel(weightsPath, configPath)

# configure parameters
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

########################################

while True:
    success, img = cap.read()
    
    img = cv2.resize(img, (800, 800))
    
    if img is None: # error
        print("Error: Empty frame")
        continue  # Skip processing this frame

    
    # send img to model
    classIds, confs, bbox = net.detect(img, confThreshold = 0.5)
    print(classIds, bbox)  # sanity check

    # convert detections to NumPy arrays
    classIds = np.array(classIds)
    confidence = np.array(confs)
    bbox = np.array(bbox)
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, box, color = (255,0, 0), thickness = 2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)) + "%", (box[0] + 250, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        
    # show images
    cv2.imshow("Output", img)
    cv2.waitKey(1)