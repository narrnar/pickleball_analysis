# Pickleball Video Analysis

## Introduction
This project analyzes Pickleball players in a video to measure their speed, ball shot speed, and number of shots. This project will detect players and the pickleball ball using YOLO and also utilizes CNNs to extract court keypoints.

## Output Videos
Below is a screenshot from one of the output videos:
![Pickleball analysis screenshot](/Users/darensivam/Desktop/Projects/pickleball/misc/example_output.png)

## Models Used
* YOLO v8 for Player Detection
* Fine Tuned YOLO for pickleball ball detection
* Court Key Point Extraction

* Trained YOLOV8 Model:
* Trained Pickleball court key point model:

## Training
* Pickleball ball detector with YOLO: training/pickleball_detector_training.ipynb
* Pickleball court keypoint with Pytorch: training/pickleball_court_keypoints_training.ipynb

## Requirements
* ultralytics
* pytorch
* pandas
* numpy
* opencv
















## Links

Pickleball Video: https://www.youtube.com/watch?v=C3-P4yh5yos&t=137s

Pickleball Detection Dataset: https://universe.roboflow.com/pickleball-1uztf/pickleball-uninu/dataset/1

Pickleball Court Keypoints Dataset: https://universe.roboflow.com/pickleball-ball-detection/pickleball-court-keypoints-syncz/dataset/6

Larger Pickleball Court Keypoints Dataset: https://universe.roboflow.com/aipickleballref/court-detection-bxo2j