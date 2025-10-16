import os
import cv2

def check_images():
    images_folder = "images"
    
    print("Checking images folder...")
    print(f"Folder path: {os.path.abspath(images_folder)}")
    
    if not os.path.exists(images_folder):
        print("Images folder does not exist!")
        return
    
    files = os.listdir(images_folder)
    print(f"Files found: {files}")
    
    required_files = ['happy.jpg', 'sad.jpg', 'angry.jpg', 'surprised.jpg', 'neutral.jpg', 'no_face.jpg']
    
    for filename in required_files:
        filepath = os.path.join(images_folder, filename)
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                print(f"✓ {filename} - OK (size: {img.shape})")
            else:
                print(f"✗ {filename} - Cannot read file")
        else:
            print(f"✗ {filename} - Missing")

if __name__ == "__main__":
    check_images()