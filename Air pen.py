import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw on
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Drawing parameters
drawing_color = (255, 255, 255)  # Start with white
thickness = 5
prev_x, prev_y = 0, 0

# Color options
color_options = [
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
    ("White", (255, 255, 255)),
]

def get_finger_states(hand_landmarks):
    finger_states = []
    tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    for tip_id in tip_ids:
        if tip_id == 4:  # Thumb
            if hand_landmarks.landmark[tip_id].x < hand_landmarks.landmark[tip_id - 1].x:
                finger_states.append(1)
            else:
                finger_states.append(0)
        else:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                finger_states.append(1)
            else:
                finger_states.append(0)
    return finger_states

def detect_gesture(hand_landmarks):
    finger_states = get_finger_states(hand_landmarks)
    
    # Drawing mode: Index finger up, others down
    if finger_states == [0, 1, 0, 0, 0]:
        return "draw"
    
    # Color selection mode: Index and middle fingers up, others down
    elif finger_states == [0, 1, 1, 0, 0]:
        return "color_select"
    
    # Clear canvas: All fingers up
    elif finger_states == [1, 1, 1, 1, 1]:
        return "clear"
    
    return "none"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            gesture = detect_gesture(hand_landmarks)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
            
            if gesture == "draw":
                if prev_x > 0 and prev_y > 0:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), drawing_color, thickness)
                prev_x, prev_y = x, y
            elif gesture == "color_select":
                for i, (color_name, color) in enumerate(color_options):
                    cv2.circle(image, (50, 50 + i * 50), 20, color, -1)
                    cv2.putText(image, color_name, (80, 55 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if abs(x - 50) < 30 and abs(y - (50 + i * 50)) < 30:
                        drawing_color = color
            elif gesture == "clear":
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                prev_x, prev_y = 0, 0
            
            cv2.putText(image, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine the canvas with the camera image
    combined_image = cv2.addWeighted(image, 0.7, canvas, 0.3, 0)
    cv2.imshow('Gesture Controlled Pen', combined_image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()