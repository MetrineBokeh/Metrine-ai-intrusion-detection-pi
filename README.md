# ai-intrusion-detection-pi
Face Recognition with Dlib and YOLO | Real-Time Surveillance SystemThis project implements a low-cost, AI-based facial recognition system for intrusion detection using Raspberry Pi. The system uses live camera feed to detect known and unknown faces using two different models—Dlib and YOLOv4—and compares their performance using precision, recall, and F1-score metrics.

# Requirements
Make sure you have the following installed:
Python 3.7 or later
pip
Raspberry Pi OS or any Linux/Windows with OpenCV support
Raspberry Pi Camera Module (for live capture)


# Create and Activate a Virtual Environment
# Create a virtual environment
python -m venv venv1
# Activate it on Windows
venv1\Scripts\activate
# Or on Linux/macOS
source venv1/bin/activate

# Install the necessary libraries:
pip install -r requirements.txt
If you don’t have a requirements.txt, generate one by:
pip freeze > requirements.txt

# Dataset
This repo includes a small demo image database in the /database/ folder.
To get the full dataset used in this project, download it from Kaggle:
#Once downloaded, place the images into:
/home/yourname/Desktop/myfiles/APP_NEW/database/

# Running the code For Dlib Face Recognition:
python app_recognition_stats_dlib.py
# For Yolo
python app_recognition_stats_yolov4.py



# You can perform actions such as:, 
Live camera face detection, 
Face recognition, Capture and save images, 
Train new faces, Monitor precision, 
recall, F1-score

# Features include 
Features
Real-time facial recognition
Intrusion alerts for unknown faces
Camera display integration
Model comparison (Dlib vs YOLO)
Precision, Recall, F1-score analysis
Built using Python, OpenCV, Dlib, and YOLOv4


# Repository Stucture
├── app_recognition_stats_dlib.py
├── app_recognition_stats_yolov4.py
├── database/
│   ├── Metrine1.jpg
│   ├── Robby1.jpg
│   └── ...
├── README.md
└── requirements.txt

# Written by 
Metrine Osiemo.
Master Student – Data Science
Minnesota State University, Mankato


