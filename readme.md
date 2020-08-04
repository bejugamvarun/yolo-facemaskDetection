# Face Mask Detector
* Implemented You Only Look Once (YOLO) algorithm in tensorflow 2.0
* Used Darknet yolo weights trained on COCO dataset which has 80 classes
* Implemented Transfer learning on FaceMask dataset to detect persons with masks in a live video stream


## YOLO Complete layer architecture
![yolo image](./yolo%20complete.jpg)


## Usage

* Download the pre-trained yolo weights from https://pjreddie.com/media/files/yolov3.weights or
```bash
git clone https://github.com/varun0603/yolo-facemaskDetection.git
cd ./yolo-facemaskDetection
wget https://pjreddie.com/media/files/yolov3.weights -O weights/yolov3.weights
```
