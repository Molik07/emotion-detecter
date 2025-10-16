import cv2
import numpy as np
import os

def create_sample_images():
    # Create images directory if it doesn't exist
    images_dir = "images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    # Create simple colored images with text
    images_data = {
        'happy.jpg': (0, 255, 0, 'HAPPY! :)'),
        'sad.jpg': (255, 0, 0, 'SAD :('),
        'angry.jpg': (0, 0, 255, 'ANGRY >:('),
        'surprised.jpg': (0, 255, 255, 'SURPRISED :O'),
        'neutral.jpg': (128, 128, 128, 'NEUTRAL :|'),
        'no_face.jpg': (255, 255, 255, 'NO FACE DETECTED')
    }
    
    for filename, (b, g, r, text) in images_data.items():
        # Create image
        img = np.full((400, 600, 3), (b, g, r), dtype=np.uint8)
        
        # Add text
        cv2.putText(img, text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        # Save image
        cv2.imwrite(os.path.join(images_dir, filename), img)
        print(f"Created {filename}")

if __name__ == "__main__":
    create_sample_images()
    print("All images created! You can now run the main program.")