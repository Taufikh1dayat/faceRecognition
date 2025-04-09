import cv2
import mediapipe as mp
import numpy as np

def detect_face_orientation():
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    canvas = None
    prev_x, prev_y = None, None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face Detection
        results = face_detection.process(rgb_frame)
        
        # Hand Detection
        hand_results = hands.process(rgb_frame)
        
        # Check if a face is detected
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, "Muka", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Belakang", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Process hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, "Tangan Terdeteksi", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Extract finger tip positions
                index_finger_tip = hand_landmarks.landmark[8]  # Ujung jari telunjuk
                h, w, _ = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                
                # Check if all five fingers are extended (to clear canvas)
                finger_tips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # Thumb to pinky
                extended_fingers = [tip.y < hand_landmarks.landmark[i - 2].y for i, tip in zip([4, 8, 12, 16, 20], finger_tips)]
                
                if all(extended_fingers):
                    canvas = np.zeros_like(frame)  # Clear the canvas
                else:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                    prev_x, prev_y = x, y
        else:
            prev_x, prev_y = None, None
        
        # Combine frame and canvas
        frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        
        cv2.imshow('Face and Hand Detection with Drawing', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face_orientation()