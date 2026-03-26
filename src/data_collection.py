import cv2
import mediapipe as mp
import csv
import os

SIGN_NAME = input("Enter sign name: ")

# Create folder if not exists
DATA_PATH = f"data/raw/{SIGN_NAME}"
os.makedirs(DATA_PATH, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Two-hand detection
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

sample_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        landmarks = []

        # Collect landmarks from both hands
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # If only one hand detected, pad zeros
        if len(landmarks) < 126:
            landmarks.extend([0] * (126 - len(landmarks)))

        # Save CSV file
        file_path = os.path.join(DATA_PATH, f"{sample_count}.csv")
        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(landmarks)

        sample_count += 1

        # Draw both hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(frame, f"Samples: {sample_count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("2-Hand Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()