# YOLO-Object-Detection

![Object Detection](https://drive.google.com/uc?export=view&id=1HJNFN2Z_g_Wcv4nGerVkOaDZg2Nec-JD)

You need to download YOLO weights as well as the config file which are provided in this [link](https://pjreddie.com/darknet/yolo/), after that put those files in the "Data" folder

To run the script, you simply just got to the directory and run command below in the command prompt
- To perform object detection on image:
```
python object_detection_using_yolo.py --image "Images and Videos/Image 1.jpg"
```
- To perform object detection on video:
```
python video_object_detection_using_yolo.py --video "Thailand traffic footage for object recognition.mp4"
```
- To perform object detection in real time:
```
python real_time_object_detection_using_yolo.py
```
