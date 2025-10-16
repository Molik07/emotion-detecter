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
        self.confidence_threshold = 0.2  # Lower from 0.3 to 0.2 for more sensitive detection
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # Only analyze every 0.5 seconds for performance
        
        # Face detection parameters for better tracking
        self.scale_factor = 1.1  # Smaller value = more thorough detection
        self.min_neighbors = 4   # Reduced for better sensitivity
        self.min_face_size = (60, 60)  # Minimum face size
        
        self.sad_boost = 1.2  # Boost sad emotion confidence
        
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
    
    def boost_emotion_confidence(self, emotions, dominant_emotion):
        """
        Boost confidence for emotions that are commonly under-detected
        """
        boosted_emotions = emotions.copy()
        
        # Boost sad emotion detection
        if 'sad' in boosted_emotions:
            boosted_emotions['sad'] *= 1.3  # Increase sad confidence by 30%
        
        # Boost angry detection (often confused with sad)
        if 'angry' in boosted_emotions:
            boosted_emotions['angry'] *= 1.1
        
        # Find new dominant emotion after boosting
        new_dominant = max(boosted_emotions, key=boosted_emotions.get)
        
        return boosted_emotions, new_dominant