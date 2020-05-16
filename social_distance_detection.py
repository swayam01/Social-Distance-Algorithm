import numpy as np
import argparse
import sys
import cv2
from math import pow, sqrt


# Parse the arguments from command line
arg = argparse.ArgumentParser(description='Social distance detection')

arg.add_argument('-v', '--video', type = str, default = '', help = 'Video file path. If no path is given, video is captured using device.')

arg.add_argument('-m', '--model', required = True, help = "Path to the pretrained model.")

arg.add_argument('-p', '--prototxt', required = True, help = 'Prototxt of the model.')

arg.add_argument('-l', '--labels', required = True, help = 'Labels of the dataset.')

arg.add_argument('-c', '--confidence', type = float, default = 0.2, help='Set confidence for detecting objects')

args = vars(arg.parse_args())


labels = [line.strip() for line in open(args['labels'])]

# Generate random bounding box bounding_box_color for each label
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))


# Load model
print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("\nStreaming video using device...\n")


# Capture video from file or through device
if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0)


frame_no = 0

while cap.isOpened():

    frame_no = frame_no+1

    # Capture one frame after another
    ret, frame = cap.read()

    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    network.setInput(blob)
    detections = network.forward()

    pos_dict = dict()
    coordinates = dict()

    # Focal length
    F = 615

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:

            class_id = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Filtering only persons detected in the frame. Class Id of 'person' is 15
            if class_id == 15.00:

                # Draw bounding box for the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 2)

                label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                print("{}".format(label))


                coordinates[i] = (startX, startY, endX, endY)

                # Mid point of bounding box
                x_mid = round((startX+endX)/2,4)
                y_mid = round((startY+endY)/2,4)

                height = round(endY-startY,4)

                # Distance from camera based on triangle similarity
                distance = (165 * F)/height
                print("Distance(cm):{dist}\n".format(dist=distance))

                # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                x_mid_cm = (x_mid * distance) / F
                y_mid_cm = (y_mid * distance) / F
                pos_dict[i] = (x_mid_cm,y_mid_cm,distance)

    # Distance between every object detected in a frame
    close_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # Check if distance less than 2 metres or 200 centimetres
                if dist < 200:
                    close_objects.add(i)
                    close_objects.add(j)

    for i in pos_dict.keys():
        if i in close_objects:
            COLOR = (0,0,255)
        else:
            COLOR = (0,255,0)
        (startX, startY, endX, endY) = coordinates[i]

        cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        # Convert cms to feet
        cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
cv2.destroyAllWindows()
