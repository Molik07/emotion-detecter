import cv2
import numpy as np

class FaceTracer:
    def __init__(self):
        """Initialize OpenCV face and feature detectors"""
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def draw_face_landmarks(self, frame):
        """Draw face detection and feature tracking"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Draw main face outline (white rectangle)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            # Draw face oval overlay
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 255), 2)
            
            # Draw face center point
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
            # Create face region for feature detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect and draw eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                # Eye rectangles
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                # Eye centers
                eye_center = (ex + ew//2, ey + eh//2)
                cv2.circle(roi_color, eye_center, 3, (0, 255, 0), -1)
                
                # Draw eye "iris" tracking
                cv2.circle(roi_color, eye_center, ew//4, (255, 0, 0), 2)
            
            # Detect smile/mouth
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                # Draw smile center
                smile_center = (sx + sw//2, sy + sh//2)
                cv2.circle(roi_color, smile_center, 3, (0, 0, 255), -1)
            
            # Draw facial feature connection lines
            self.draw_face_connections(frame, x, y, w, h, eyes)
            
            # Draw face measurement grid
            self.draw_measurement_grid(frame, x, y, w, h)
        
        return frame
    
    def draw_face_connections(self, frame, face_x, face_y, face_w, face_h, eyes):
        """Draw connecting lines between facial features"""
        face_center = (face_x + face_w//2, face_y + face_h//2)
        
        # Draw lines from face center to corners
        corners = [
            (face_x, face_y),
            (face_x + face_w, face_y),
            (face_x, face_y + face_h),
            (face_x + face_w, face_y + face_h)
        ]
        
        for corner in corners:
            cv2.line(frame, face_center, corner, (100, 100, 100), 1)
        
        # Connect eyes if detected
        if len(eyes) >= 2:
            eye1_center = (face_x + eyes[0][0] + eyes[0][2]//2, face_y + eyes[0][1] + eyes[0][3]//2)
            eye2_center = (face_x + eyes[1][0] + eyes[1][2]//2, face_y + eyes[1][1] + eyes[1][3]//2)
            cv2.line(frame, eye1_center, eye2_center, (255, 255, 0), 2)
            
            # Connect eyes to face center
            cv2.line(frame, eye1_center, face_center, (255, 100, 100), 1)
            cv2.line(frame, eye2_center, face_center, (255, 100, 100), 1)
    
    def draw_measurement_grid(self, frame, x, y, w, h):
        """Draw measurement grid on face"""
        # Vertical lines
        for i in [1, 2]:
            line_x = x + (w * i // 3)
            cv2.line(frame, (line_x, y), (line_x, y + h), (80, 80, 80), 1)
        
        # Horizontal lines
        for i in [1, 2]:
            line_y = y + (h * i // 3)
            cv2.line(frame, (x, line_y), (x + w, line_y), (80, 80, 80), 1)
        
        # Center cross
        center_x, center_y = x + w//2, y + h//2
        cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 0), 2)
    
    def draw_simple_face_trace(self, frame):
        """Simplified tracing - just boxes and points"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Simple face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Corner markers
            corner_size = 20
            corners = [
                (x, y), (x+w, y), (x, y+h), (x+w, y+h)
            ]
            
            for corner_x, corner_y in corners:
                cv2.line(frame, (corner_x-corner_size//2, corner_y), (corner_x+corner_size//2, corner_y), (255, 0, 255), 3)
                cv2.line(frame, (corner_x, corner_y-corner_size//2), (corner_x, corner_y+corner_size//2), (255, 0, 255), 3)
        
        return frame
    
    def draw_basic_face_box(self, frame):
        """Most basic - just bounding box"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'FACE DETECTED', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame