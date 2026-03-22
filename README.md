# Real-Time 3D Face Tracking (MediaPipe + OpenCV)

This project performs real-time 3D face tracking using MediaPipe Face Mesh and OpenCV.
It detects facial landmarks and estimates depth to create a 3D face representation.

---

## Features

* Real-time face detection
* 468 facial landmarks
* Depth (Z-axis) estimation
* Face mesh rendering
* Webcam integration
* Fast processing

---

## Tech Stack

* Python
* OpenCV
* MediaPipe
* NumPy

---

## Project Structure

* face_3d.py
* requirements.txt
* README.md
* screenshots/

---

## Installation

Clone the repository:

git clone https://github.com/your-username/3d-face-tracking.git

Go to the folder:

cd 3d-face-tracking

Install dependencies:

pip install -r requirements.txt

Or install manually:

pip install opencv-python mediapipe numpy

---

## Usage

Run the script:

python face_3d.py

* Webcam will start
* Face mesh will appear
* Landmarks track face in real time
* Press **Q** to exit

---

## How It Works

* Webcam captures frames
* MediaPipe detects facial landmarks
* Each point has (x, y, z) coordinates
* Z value gives depth
* Mesh is drawn on the face
* Updates happen in real time

---

## Applications

* AR filters
* Face animation
* Head pose tracking
* Human-computer interaction

---

## Future Improvements

* Head pose estimation
* Eye blink detection
* Expression recognition
* AR overlays

---

## Learning

* 3D landmark basics
* Depth estimation
* Real-time tracking
* MediaPipe usage
