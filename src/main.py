# File: /facial-expression-detector/facial-expression-detector/src/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from emotion_detector import EmotionDetector
from image_manager import ImageManager
from face_tracer import FaceTracer
import time

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create instances
    emotion_detector = EmotionDetector()
    face_tracer = FaceTracer()
    images_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
    image_manager = ImageManager(images_folder)
    
    current_emotion = 'neutral'
    show_emotion_text = False
    show_face_trace = False
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect emotion
        detected_emotion = emotion_detector.detect_emotion(frame)
        
        if detected_emotion is not None:
            current_emotion = detected_emotion
        else:
            current_emotion = 'no_face'
        
        # Add face tracing if toggled ON
        if show_face_trace:
            frame = face_tracer.draw_face_landmarks(frame)
        
        # Display corresponding emotion image
        image_manager.display_image(current_emotion)
        
        # Add emotion text if toggled ON
        if show_emotion_text:
            emotion_text = f"Emotion: {current_emotion.upper()}"
            cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
        
        # Show webcam feed
        cv2.imshow('Webcam Feed', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_emotion_text = not show_emotion_text
        elif key == ord('f'):
            show_face_trace = not show_face_trace
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Explanation:
# The main function initializes the webcam and creates instances of EmotionDetector and ImageManager.
# It enters a loop where it captures frames from the webcam, detects emotions, and displays the corresponding image.
# The loop continues until the 'q' key is pressed, at which point it releases the webcam and closes all OpenCV windows.

import cv2
import numpy as np
from deepface import DeepFace
import time

class EmotionDetector:
    def __init__(self):
        """
        Initialize the EmotionDetector class with improved settings for smoother detection.
        """
        # Load the face detection model with better parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Smoothing parameters
        self.last_emotion = 'neutral'
        self.emotion_history = []
        self.history_size = 5  # Number of frames to consider for smoothing
        self.confidence_threshold = 0.3
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # Only analyze every 0.5 seconds for performance
        
        # Face detection parameters for better tracking
        self.scale_factor = 1.1  # Smaller value = more thorough detection
        self.min_neighbors = 4   # Reduced for better sensitivity
        self.min_face_size = (60, 60)  # Minimum face size
        
    def detect_emotion(self, frame):
        """
        Detect emotion from the given frame with smoothing.
        
        Args:
            frame: Input video frame from webcam
            
        Returns:
            str: Smoothed detected emotion or None if no face detected
        """
        try:
            current_time = time.time()
            
            # First, do fast face detection every frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Improve face detection with better parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If no face detected, return None immediately
            if len(faces) == 0:
                self.emotion_history.clear()  # Clear history when no face
                return None
            
            # Only run DeepFace analysis periodically to improve performance
            if current_time - self.last_analysis_time < self.analysis_interval:
                return self.get_smoothed_emotion()
            
            self.last_analysis_time = current_time
            
            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Extract face region with some padding
            padding = 20
            face_region = frame[max(0, y-padding):y+h+padding, 
                              max(0, x-padding):x+w+padding]
            
            if face_region.size == 0:
                return self.get_smoothed_emotion()
            
            # Convert to RGB for DeepFace
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Analyze emotion with relaxed settings
            result = DeepFace.analyze(
                rgb_face, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True  # Suppress DeepFace logs
            )
            
            # Handle DeepFace result
            if isinstance(result, list):
                result = result[0]
            
            # Get emotion with confidence
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion] / 100.0  # Convert to 0-1 range
            
            # Only update if confidence is high enough
            if confidence > self.confidence_threshold:
                # Map DeepFace emotions to our image names
                emotion_mapping = {
                    'happy': 'happy',
                    'sad': 'sad', 
                    'angry': 'angry',
                    'surprise': 'surprised',
                    'neutral': 'neutral',
                    'fear': 'surprised',
                    'disgust': 'angry'
                }
                
                mapped_emotion = emotion_mapping.get(dominant_emotion, 'neutral')
                self.add_to_history(mapped_emotion)
                
            return self.get_smoothed_emotion()
            
        except Exception as e:
            # Return last known good emotion on error
            return self.get_smoothed_emotion()
    
    def add_to_history(self, emotion):
        """Add emotion to history for smoothing"""
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
    
    def get_smoothed_emotion(self):
        """Get smoothed emotion based on recent history"""
        if not self.emotion_history:
            return self.last_emotion
        
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return most frequent emotion
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        self.last_emotion = smoothed_emotion
        return smoothed_emotion

import cv2
import os
import numpy as np

class ImageManager:
    def __init__(self, image_folder):
        """
        Initialize the ImageManager with the path to the image folder.
        
        Args:
            image_folder (str): Path to the folder containing emotion images
        """
        self.image_folder = image_folder
        print(f"ImageManager: Looking for images in {os.path.abspath(image_folder)}")
        
        # Dictionary to store loaded images
        self.images = {}
        self.load_all_images()
        
    def load_all_images(self):
        """Load all emotion images from the folder"""
        # Define the required image files
        required_images = {
            'happy': 'happy.jpg',
            'sad': 'sad.jpg', 
            'angry': 'angry.jpg',
            'surprised': 'surprised.jpg',
            'neutral': 'neutral.jpg',
            'no_face': 'no_face.jpg'
        }
        
        # Check if folder exists
        if not os.path.exists(self.image_folder):
            print(f"Images folder {self.image_folder} does not exist!")
            os.makedirs(self.image_folder, exist_ok=True)
        
        for emotion, filename in required_images.items():
            image_path = os.path.join(self.image_folder, filename)
            
            print(f"Trying to load: {image_path}")
            
            if os.path.exists(image_path):
                # Load the actual image from your folder
                image = cv2.imread(image_path)
                if image is not None:
                    self.images[emotion] = image
                    print(f"✓ Successfully loaded {filename} (shape: {image.shape})")
                else:
                    print(f"✗ Failed to load {filename} - file corrupted")
                    self.images[emotion] = self.create_placeholder_image(emotion)
            else:
                print(f"✗ {filename} not found at {image_path}")
                self.images[emotion] = self.create_placeholder_image(emotion)

    def create_placeholder_image(self, emotion):
        """Create a simple placeholder image if the actual image is missing"""
        
        # Color mapping for emotions (B, G, R format for OpenCV)
        colors = {
            'happy': (0, 255, 0),        # Green
            'sad': (255, 0, 0),          # Blue
            'angry': (0, 0, 255),        # Red
            'surprised': (0, 255, 255),  # Yellow
            'neutral': (128, 128, 128),  # Gray
            'no_face': (200, 200, 200)   # Light Gray
        }
        
        color = colors.get(emotion, (128, 128, 128))
        img = np.full((300, 300, 3), color, dtype=np.uint8)
        
        # Add simple text
        text = f"{emotion.upper()}"
        cv2.putText(img, text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return img

    def get_image_path(self, emotion):
        """
        Get the file path for a given emotion.
        
        Args:
            emotion (str): The detected emotion
            
        Returns:
            str: Path to the corresponding image file
        """
        if emotion in self.images:
            return emotion
        else:
            return 'neutral'  # Default to neutral if emotion not recognized

    def display_image(self, emotion):
        """
        Display the image corresponding to the given emotion.
        
        Args:
            emotion (str): The emotion for which to display the image
        """
        if emotion in self.images:
            # Get the original image
            original_image = self.images[emotion]
            
            # Make a copy to avoid modifying the original
            display_image = original_image.copy()
            
            # Resize image to fit window nicely (optional)
            height, width = display_image.shape[:2]
            
            # If image is too large, resize it
            max_size = 600
            if height > max_size or width > max_size:
                if height > width:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                else:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                
                display_image = cv2.resize(display_image, (new_width, new_height))
            
            # Display the image without any text overlay
            cv2.imshow('Current Emotion', display_image)
        else:
            # Display neutral image as default
            if 'neutral' in self.images:
                cv2.imshow('Current Emotion', self.images['neutral'])
            else:
                # Create a simple default image
                default_img = np.full((300, 300, 3), (128, 128, 128), dtype=np.uint8)
                cv2.putText(default_img, "NO IMAGE", (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Current Emotion', default_img)

from imutils import face_utils
import dlib
import cv2

class FaceTracer:
    def __init__(self):
        """
        Initialize the FaceTracer class with dlib's face detector and predictor.
        """
        # Load dlib's face detector
        self.detector = dlib.get_frontal_face_detector()
        
        # Load dlib's shape predictor for facial landmarks
        predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update path as needed
        self.predictor = dlib.shape_predictor(predictor_path)

    def draw_face_landmarks(self, frame):
        """
        Detect faces and draw facial landmarks on the frame.
        
        Args:
            frame: Input video frame from webcam
            
        Returns:
            frame: Output video frame with facial landmarks drawn
        """
        # Convert the frame to grayscale for dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = self.detector(gray)
        
        # Loop over the detected faces
        for face in faces:
            # Get the landmarks/parts for the face in box dlib.rectangle format
            shape = self.predictor(gray, face)
            
            # Convert the landmarks to a NumPy array
            landmarks = face_utils.shape_to_np(shape)
            
            # Draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw the facial landmarks
            for (i, (x, y)) in enumerate(landmarks):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        return frame