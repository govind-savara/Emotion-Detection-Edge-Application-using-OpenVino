![Intel® Edge AI Foundation Course](https://img.shields.io/badge/Udacity-Intel%C2%AE%20Edge%20AI%20Foundation%20Course-blue?logo=Udacity&color=33bbff&style=flat)

[image1]: ./imgs/Output_emotions_img.jpg "Inference on Image"
[image2]: ./imgs/Output_emotions_video.jpg "Inference on Videos"

# Emotion Detection Edge Application using OpenVino

This Emotion Detection application can be used to detect the emotions of the faces on an image or video stream using Intel® hardware and software tools.

**Inference on Image**

![Inference on Image][image1]

**Inference on Video**

![Inference on Video][image2]

## Models

I have used two models:
 - [face-detection-adas-0001](https://docs.openvinotoolkit.org/2019_R1/_face_detection_adas_0001_description_face_detection_adas_0001.html)
 - [emotions-recognition-retail-0003](https://docs.openvinotoolkit.org/2019_R1/_emotions_recognition_retail_0003_description_emotions_recognition_retail_0003.html)

First used the Face Detection Model to detect the faces and then pipelined emotions model to detect the emotions.

The Emotions Recognition model recognises five emotions ('neutral', 'happy', 'sad', 'surprise', 'anger').

## Requirements

### Software

*  [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download?)
*  OR Udacity classroom workspace for the related course

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Utilize the classroom workspace, or refer to the relevant instructions for your operating system for this step.

- [Linux/Ubuntu](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
- [Mac](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)
- [Windows](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)

## Run the Program 

### Command Line Arguments

 - `-i`  : The location of the input file or "CAM" for the camera streaming
 - `-d`  : The device name, if not "CPU"
 - `-ct` : The confidence threshold to use with the bounding boxes for face detection, default value is 0.5
 
### Command to run the program
 For input image or video 
 
 ```python app.py -i input/image/or/video/path -ct 0.6```
 
 For camera streaming
 
 ```python app.py -i CAM -ct 0.7```
 
## Future Works 
 - implement the Server Communications for this application
 - calculate the overall emotions in a videos to determine the video type
 - Analysis of predicted emotions and time duration on the video
 - include more emotion types in the video
 - This model can be used for emotions detection in films for the film certification