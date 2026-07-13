import os
import dlib
import numpy as np
import cv2

class GeometryEstimator:
    def __init__(self, predictor_path='C:/tmp/ckpt/shape_predictor_68_face_landmarks.dat'):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, img_np):
        """
        img_np: RGB image numpy array (H, W, 3)
        Returns: list of 68 (x, y) coordinates or None if no face detected
        """
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) == 0:
            return None
        # Use first detected face
        shape = self.predictor(gray, rects[0])
        landmarks = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
        return landmarks

    def calculate_geometry(self, landmarks):
        """
        landmarks: numpy array (68, 2)
        Returns a dictionary of geometric features.
        """
        if landmarks is None:
            return None
            
        def dist(p1, p2):
            return np.linalg.norm(p1 - p2)

        # 1. Face Width (L0 to L16)
        face_width = dist(landmarks[0], landmarks[16])

        # 2. Face Height (L8 to L27)
        face_height = dist(landmarks[8], landmarks[27])

        # 3. Width/Height Ratio
        wh_ratio = face_width / face_height if face_height > 0 else 0.0

        # 4. Jaw Width (L4 to L12)
        jaw_width = dist(landmarks[4], landmarks[12])

        # 5. Cheek Width (Cheekbone Width) (L2 to L14)
        cheek_width = dist(landmarks[2], landmarks[14])

        # 6. Interocular Distance (Eye spacing - inner corners L39 to L42)
        interocular_distance = dist(landmarks[39], landmarks[42])

        # 7. Nose Width (L31 to L35)
        nose_width = dist(landmarks[31], landmarks[35])

        # 8. Mouth Width (L48 to L54)
        mouth_width = dist(landmarks[48], landmarks[54])

        # 9. Jaw Angle at chin (L8) between vector (4->8) and (12->8)
        v1 = landmarks[4] - landmarks[8]
        v2 = landmarks[12] - landmarks[8]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        jaw_angle = np.degrees(np.arccos(cos_theta))

        return {
            'Face Width': float(face_width),
            'Face Height': float(face_height),
            'Width/Height Ratio': float(wh_ratio),
            'Jaw Width': float(jaw_width),
            'Cheek Width': float(cheek_width),
            'Interocular Distance': float(interocular_distance),
            'Nose Width': float(nose_width),
            'Mouth Width': float(mouth_width),
            'Jaw Angle': float(jaw_angle)
        }

    def estimate_image_geometry(self, img_np):
        """
        Direct helper to estimate geometry from image array
        """
        landmarks = self.get_landmarks(img_np)
        return self.calculate_geometry(landmarks)
