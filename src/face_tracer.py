import cv2


class FaceTracer:
    def __init__(self):
        """Initialize OpenCV face and feature detectors (no dlib/mediapipe)."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

    def draw_face_landmarks(self, frame):
        """Detect face, draw box, eyes, smile, connections and grid."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Main face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # Approx face oval
            center = (x + w // 2, y + h // 2)
            axes = (w // 2, h // 2)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 255), 2)

            # Eyes and smile within ROI
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(20, 20))
            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                c = (ex + ew // 2, ey + eh // 2)
                eye_centers.append((x + c[0], y + c[1]))
                cv2.circle(roi_color, c, 3, (0, 255, 0), -1)
                cv2.circle(roi_color, c, max(2, ew // 4), (255, 0, 0), 1)

            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

            self.draw_face_connections(frame, x, y, w, h, eye_centers)
            self.draw_measurement_grid(frame, x, y, w, h)

        return frame

    def draw_face_connections(self, frame, face_x, face_y, face_w, face_h, eye_centers):
        """Simple feature connections for visualization."""
        face_center = (face_x + face_w // 2, face_y + face_h // 2)
        corners = [
            (face_x, face_y),
            (face_x + face_w, face_y),
            (face_x, face_y + face_h),
            (face_x + face_w, face_y + face_h),
        ]
        for corner in corners:
            cv2.line(frame, face_center, corner, (100, 100, 100), 1)

        if len(eye_centers) >= 2:
            cv2.line(frame, eye_centers[0], eye_centers[1], (255, 255, 0), 2)
            cv2.line(frame, eye_centers[0], face_center, (255, 100, 100), 1)
            cv2.line(frame, eye_centers[1], face_center, (255, 100, 100), 1)

    def draw_measurement_grid(self, frame, x, y, w, h):
        """Rule-of-thirds grid + center cross inside face box."""
        for i in (1, 2):
            lx = x + (w * i // 3)
            ly = y + (h * i // 3)
            cv2.line(frame, (lx, y), (lx, y + h), (80, 80, 80), 1)
            cv2.line(frame, (x, ly), (x + w, ly), (80, 80, 80), 1)

        cx, cy = x + w // 2, y + h // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 2)

    # Optional simpler modes if you toggle to use them later:
    def draw_simple_face_trace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return frame

    def draw_basic_face_box(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return frame