<<<<<<< HEAD
# Facial Expression Detector

This project is a real-time facial expression detection application that uses your webcam to analyze your emotions and display corresponding images based on the detected emotion. The application utilizes the OpenCV library for video capture and the DeepFace library for emotion detection.

## Project Structure

```
facial-expression-detector
├── src
│   ├── main.py              # Entry point of the application
│   ├── emotion_detector.py   # Contains the EmotionDetector class
│   └── image_manager.py      # Contains the ImageManager class
├── images
│   ├── happy.jpg            # Image for happy emotion
│   ├── sad.jpg              # Image for sad emotion
│   ├── angry.jpg            # Image for angry emotion
│   ├── surprised.jpg        # Image for surprised emotion
│   ├── neutral.jpg          # Image for neutral emotion
│   └── no_face.jpg          # Image when no face is detected
├── requirements.txt         # Lists project dependencies
└── README.md                # Project documentation
```

## Setup Instructions

To run this project, you need to install the required dependencies. You can do this by creating a virtual environment and installing the packages listed in `requirements.txt`. Follow these steps:

1. **Clone the repository** (if applicable):
   ```
   git clone <repository-url>
   cd facial-expression-detector
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```

## Running the Program

To run the facial expression detector, execute the following command in your terminal:

```
python src/main.py
```

Make sure your webcam is connected and accessible. The program will open a window displaying the video feed from your webcam. As it detects your facial expressions, it will change the displayed image according to the detected emotion.

## Dependencies

The following libraries are required for this project:

- OpenCV: For capturing video from the webcam and displaying images.
- DeepFace: For emotion detection from the captured video frames.
- Pillow: For image handling.

You can find the specific versions of these libraries in the `requirements.txt` file.

## Notes

- Ensure that the images (happy.jpg, sad.jpg, angry.jpg, surprised.jpg, neutral.jpg, no_face.jpg) are placed in the `images` directory as specified in the project structure.
- The application continuously updates the displayed image based on your facial expression, so make sure to test different emotions for the best experience.
=======
# emotion-detecter
detects your emotion through your webcam and show an emotion image accordingly
>>>>>>> 33c0222a1eaf64716c014c27a153b66adb53110e
