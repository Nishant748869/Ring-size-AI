import math
import cv2
import numpy as np

# Simple calibration: Pixels to Centimeters
# Assumes a fixed distance from the webcam.
# Adjust this value to calibrate the measurements!
PIXELS_TO_CM = 0.05 

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
# Landmark indices near the middle of each finger:
# Thumb: IP (3), Others: PIP (6, 10, 14, 18)
FINGER_MIDDLE_JOINTS = [3, 6, 10, 14, 18]

def get_finger_width(frame, hand_landmarks, finger_idx):
    """
    Estimates the width of a finger in pixels near the middle joint.
    Scans perpendicularly to the finger direction to find image edges.
    """
    h, w, _ = frame.shape
    
    # Get the middle joint landmark
    joint_idx = FINGER_MIDDLE_JOINTS[finger_idx]
    joint = hand_landmarks.landmark[joint_idx]
    cx, cy = int(joint.x * w), int(joint.y * h)
    
    # Get the joint above and below to determine finger direction
    if finger_idx == 0: # Thumb
        upper_idx, lower_idx = 4, 2 # Tip and MCP
    else:
        upper_idx, lower_idx = joint_idx + 1, joint_idx - 1 # DIP and MCP
        
    upper = hand_landmarks.landmark[upper_idx]
    lower = hand_landmarks.landmark[lower_idx]
    
    ux, uy = int(upper.x * w), int(upper.y * h)
    lx, ly = int(lower.x * w), int(lower.y * h)
    
    # Direction vector of the finger
    dx = ux - lx
    dy = uy - ly
    
    # Normal (perpendicular) vector
    length = math.hypot(dx, dy)
    if length == 0:
        return 0, None, None
        
    px = -dy / length
    py = dx / length
    
    # Convert image to grayscale for edge scanning
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply blur to smooth out noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    try:
        base_intensity = int(gray[cy, cx])
    except IndexError:
        return 0, None, None
    
    # Limit how far we scan to avoid picking up background objects
    # We estimate max width based on the distance between PIP and MCP
    max_scan_dist = int(math.hypot(cx - lx, cy - ly) * 1.5)
    max_scan_dist = max(20, min(max_scan_dist, 60)) # clamp between 20 and 60 pixels
    
    def scan_edge(step_x, step_y):
        for i in range(1, max_scan_dist):
            nx = int(cx + step_x * i)
            ny = int(cy + step_y * i)
            
            # Check image boundaries
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                return nx, ny
            
            intensity = int(gray[ny, nx])
            
            # If intensity drops or rises significantly, we probably hit the edge of the finger
            if abs(intensity - base_intensity) > 25: 
                return nx, ny
                
        # If no edge found within limit, return the max distance point
        return int(cx + step_x * max_scan_dist), int(cy + step_y * max_scan_dist)
        
    # Scan positive perpendicular direction
    edge1_x, edge1_y = scan_edge(px, py)
    # Scan negative perpendicular direction
    edge2_x, edge2_y = scan_edge(-px, -py)
    
    width_pixels = math.hypot(edge1_x - edge2_x, edge1_y - edge2_y)
    
    return width_pixels, (edge1_x, edge1_y), (edge2_x, edge2_y)

def calculate_circumference(width_pixels):
    """
    Calculates the circumference in cm assuming a circular cross-section: C = pi * d
    """
    diameter_cm = width_pixels * PIXELS_TO_CM
    circumference = math.pi * diameter_cm
    return circumference

def process_measurements(frame, hand_landmarks, extended_fingers):
    """
    Calculates circumferences for extended fingers and draws visual measurement lines.
    Returns a dictionary of circumference results and the modified frame.
    """
    results = {}
    
    num_extended = sum(extended_fingers)
    if num_extended == 0:
        return results, frame
        
    for i, is_extended in enumerate(extended_fingers):
        if is_extended:
            width_px, p1, p2 = get_finger_width(frame, hand_landmarks, i)
            if width_px > 0:
                circ = calculate_circumference(width_px)
                results[FINGER_NAMES[i]] = circ
                
                # Draw measurement line (cyan color) across the finger
                if p1 and p2:
                    cv2.line(frame, p1, p2, (255, 255, 0), 2)
                    # Draw small circles at the edges
                    cv2.circle(frame, p1, 3, (0, 0, 255), -1)
                    cv2.circle(frame, p2, 3, (0, 0, 255), -1)
                    
            # If only 1 finger is extended, measure only that one and break.
            if num_extended == 1:
                break
                
    return results, frame
