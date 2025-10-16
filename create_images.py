import cv2
import numpy as np
import os

def create_sample_images():
    # Create images directory if it doesn't exist
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Create simple colored images (300x300 pixels)
    images = {
        'happy.jpg': (0, 255, 0),      # Green
        'sad.jpg': (255, 0, 0),        # Blue  
        'angry.jpg': (0, 0, 255),      # Red
        'surprised.jpg': (0, 255, 255), # Yellow
        'neutral.jpg': (128, 128, 128), # Gray
        'no_face.jpg': (255, 255, 255)  # White
    }
    
    for filename, color in images.items():
        # Create a colored image
        img = np.full((300, 300, 3), color, dtype=np.uint8)
        
        # Add text to identify the emotion
        emotion_name = filename.split('.')[0].upper()
        cv2.putText(img, emotion_name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Save the image
        cv2.imwrite(os.path.join(images_dir, filename), img)
        print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_images()
    print("All sample images created successfully!")