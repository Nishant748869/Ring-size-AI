import cv2
import time
from detector import HandDetector
from measurement import process_measurements

def main():
    print("Initializing Webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Initializing Hand Detector...")
    detector = HandDetector()
    
    pTime = 0
    
    print("Starting POC. Press 'q' to quit, 's' to save a screenshot.")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break
            
        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)
        
        # Process the frame for hand landmarks
        results = detector.process_frame(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. Determine which fingers are extended
                extended_fingers = detector.get_extended_fingers(hand_landmarks)
                
                # 2. Draw the skeletal landmarks
                detector.draw(frame, results)
                
                # 3. Measure circumferences and draw measurement lines
                measurements, frame = process_measurements(frame, hand_landmarks, extended_fingers)
                
                # 4. Display measurements overlay on screen
                y_pos = 60
                
                # If hands detected, show how many fingers are up (for debugging)
                num_up = sum(extended_fingers)
                cv2.putText(frame, f"Fingers up: {num_up}", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_pos += 30
                
                cv2.putText(frame, "Estimated Circumference:", (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30
                
                for finger_name, circ in measurements.items():
                    text = f"{finger_name}: {circ:.1f} cm"
                    cv2.putText(frame, text, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_pos += 30

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime > 0 else 0
        pTime = cTime
        
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show the video frame
        cv2.imshow("Finger Circumference Estimator POC", frame)
        
        # Keyboard interactions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Bonus: Save screenshot
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved to {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
