# Deeplearning-based-Social-Distance-Detection

Developed to combat COVID-19 outbreak by helping Government monitor areas where social distancing is not practiced and enforce stricter laws. It uses a camera to capture the video and detect people in real-time. If people are very close to each other, a red bounding box is displayed around them indicating that they are not maintainting social distance.
<br />
<br />

## Install the dependencies

```
pip install -r requirements.txt
```

## Run the code

```
python social_distance_detection.py --prototxt SSD_MobileNet_prototxt.txt --model SSD_MobileNet.caffemodel --labels class_labels.txt
or
python social_distance_detection.py --video test_sample.mp4 --prototxt SSD_MobileNet_prototxt.txt --model SSD_MobileNet.caffemodel --labels class_labels.txt
``` 


## Algorithm

Single Shot object Detection (SSD) using MobileNet and OpenCV used to detect people. A bounding box is displayed around every person detected. 

To calibrate the camera, Let us assume that a person is at a distance D (assuming distance of person from the camera is 400cm) from camera and the person's actual height is H (assuming the average height of humans is 165cm). Using the object detection model we identify the pixel height P of the person using the bounding box coordinates. Using these values, the focal length of the camera can be calculated using:

```
Eq 1: F = (P x D) / H
```
After calculating the focal length of the camera, as we continue to move camera both closer and farther away from the object, we can apply the triangle similarity to determine the distance of the object to the camera using the actual height H of the person, pixel height P of the person and focal length F of camera using formula :

```
Eq 2: D' = (H x F) / P
```
We have the x, y and z (distance of the person from camera) coordinates for every person in cm. The Euclidean distance between every person detected is calculated between the mid-point of the bounding boxes of all the people detected using the (x, y, z) coordinates. These pixel values are converted into cm using Eq 2.

If the distance between two people is less than 2 meters or 200 centimeters, a red bounding box is displayed around them indicating that they are not maintaining social distance.
