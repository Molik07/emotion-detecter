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