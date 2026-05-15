import cv2
import mediapipe as mp
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence)
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Indices for finger tips
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Little
        
        # Connections for drawing the hand skeleton
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
    def process_frame(self, frame):
        """Processes the frame and returns a legacy-compatible results object."""
        # MediaPipe Tasks API requires mp.Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = self.detector.detect(mp_image)
        
        # Wrap the result so it matches the structure expected by app.py & measurement.py
        class DummyResults:
            pass
        results = DummyResults()
        
        if detection_result.hand_landmarks:
            class DummyHandLandmarks:
                def __init__(self, lms):
                    self.landmark = lms
            results.multi_hand_landmarks = [DummyHandLandmarks(h) for h in detection_result.hand_landmarks]
        else:
            results.multi_hand_landmarks = None
            
        return results

    def get_extended_fingers(self, hand_landmarks):
        """
        Determines which fingers are extended.
        Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Little].
        """
        fingers = []
        
        def dist(lm1, lm2):
            return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)
            
        # 1. Thumb
        d_tip_mcp = dist(hand_landmarks.landmark[4], hand_landmarks.landmark[2])
        d_ip_mcp = dist(hand_landmarks.landmark[3], hand_landmarks.landmark[2])
        fingers.append(d_tip_mcp > d_ip_mcp)
            
        # 2. Other 4 fingers
        for id in range(1, 5):
            if hand_landmarks.landmark[self.tip_ids[id]].y < hand_landmarks.landmark[self.tip_ids[id] - 2].y:
                fingers.append(True)
            else:
                fingers.append(False)
                
        return fingers

    def draw(self, frame, results):
        """Draws the hand skeleton and landmarks on the frame manually."""
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw lines
                for connection in self.connections:
                    lm1 = hand_landmarks.landmark[connection[0]]
                    lm2 = hand_landmarks.landmark[connection[1]]
                    x1, y1 = int(lm1.x * w), int(lm1.y * h)
                    x2, y2 = int(lm2.x * w), int(lm2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                # Draw points
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return frame
