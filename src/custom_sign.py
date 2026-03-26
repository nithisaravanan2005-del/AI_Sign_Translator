import cv2
import mediapipe as mp
import numpy as np
import json

SIGN_NAME = input("Enter custom sign name: ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

samples = []
sample_count = 0

print("Show your custom sign...")

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                samples.append(landmarks)
                sample_count += 1

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

    cv2.imshow("Custom Sign Recording", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

# Compute average landmarks
avg_landmarks = np.mean(samples, axis=0).tolist()

# Save to JSON
try:
    with open("models/custom_signs.json", "r") as f:
        custom_signs = json.load(f)
except:
    custom_signs = {}

custom_signs[SIGN_NAME] = avg_landmarks

with open("models/custom_signs.json", "w") as f:
    json.dump(custom_signs, f, indent=4)

print(f"Custom sign '{SIGN_NAME}' saved successfully.")

print(f"Custom sign '{SIGN_NAME}' saved successfully with {len(samples)} samples.")