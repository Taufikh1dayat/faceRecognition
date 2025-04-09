# Face and Hand Detector with Drawing Feature

## Description
This is a Python program that uses OpenCV and MediaPipe to detect faces and hands in real time using a webcam. Additionally, it includes a feature that allows users to draw on the screen using their index finger. If all five fingers are extended, the canvas is cleared.

## Features
- **Face Detection**: Detects and highlights faces using a bounding box.
- **Hand Detection**: Detects hands and marks key points.
- **Drawing Feature**: Allows users to draw using their index finger.
- **Clear Canvas**: When all five fingers are extended, the canvas is cleared.

## Requirements
Ensure you have the following dependencies installed before running the program:

```sh
pip install opencv-python mediapipe numpy
```

## Usage
Run the script using Python:

```sh
python facerecogn.py
```

### Controls:
- Move your index finger to draw on the screen.
- Extend all five fingers to clear the canvas.
- Press `q` to exit the program.

## Acknowledgments
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
