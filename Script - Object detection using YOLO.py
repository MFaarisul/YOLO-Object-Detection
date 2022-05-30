import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
	help="path of the image")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(parser.parse_args())

#Get the coco classes label
with open('Data//coco_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

#Load YOLO model
net = cv2.dnn.readNet('Data//yolov3.weights', 'Data//yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread(args['image']) # Image_path
h, w, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), 0) #Shape: (1, 3, img_size, img_size)
# Channels (RGB)
# R = blob[0, 0, :, :]
# G = blob[0, 1, :, :]
# B = blob[0, 2, :, :]

net.setInput(blob)
# Detections shape (vectors of length 85):
# 4x the bounding box (centerx, centery, width, height)
# 1x box confidence
# 80x class confidence
detections = net.forward(output_layers)

class_ids = []
confidences = []
box_coordinates = []
for detection in detections:
    for vec in detection:
        # Extract class confidence
        class_confidence = vec[5:]
        class_id = np.argmax(class_confidence)
        confidence = class_confidence[class_id]
        # Try to change the threshold
        if confidence > args['confidence']:
            # Extract the coordinate of detected object in the images
            center_x = int(vec[0] * w)
            center_y = int(vec[1] * h)
            width = int(vec[2] * w)
            height = int(vec[3] * h)
            
            # Get the top left corner coordinate of the bounding boxes
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            class_ids.append(class_id)
            confidences.append(confidence)
            box_coordinates.append([left, top, width, height])

# The same object that close to each other will be counted as 1 object
# Try to change the threshold
indexes = cv2.dnn.NMSBoxes(box_coordinates, confidences, 0.5, 0.4)
# Set color for bounding boxes
colors = np.random.uniform(0, 255, (len(labels), 3))

for index in indexes:
    left, top, width, height = box_coordinates[index]
    label = labels[class_ids[index]]
    #The same object will have the same color
    color = colors[class_ids[index]]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.rectangle(img, (left, top), (left+width, top+height), color, 2)
    cv2.putText(img, label, (left, top+15), font, 0.5, (0,255,0), 2)

#Show the final image
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()