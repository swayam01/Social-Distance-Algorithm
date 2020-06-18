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
