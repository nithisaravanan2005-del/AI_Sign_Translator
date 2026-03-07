import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
from collections import deque

prediction_queue = deque(maxlen=10)

# Load trained model
model = tf.keras.models.load_model("models/sign_model.h5")

# Load label map
with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse mapping
reverse_label_map = {v: k for k, v in label_map.items()}

# Load custom signs
try:
    with open("models/custom_signs.json", "r") as f:
        custom_signs = json.load(f)
except:
    custom_signs = {}

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:

                # Check custom signs first
                matched = False
                for sign_name, saved_landmarks in custom_signs.items():
                    dist = euclidean_distance(landmarks, saved_landmarks)
                    if dist < 0.15:
                        prediction_text = f"{sign_name} (Custom)"
                        matched = True
                        break

                # If no custom match → use model
                if not matched:
                    input_data = np.array(landmarks).reshape(1, -1)
                    prediction = model.predict(input_data, verbose=0)

                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    if confidence > 0.80:
                        predicted_label = reverse_label_map[class_index]
                        prediction_queue.append(predicted_label)

                        final_prediction = max(
                            set(prediction_queue),
                            key=prediction_queue.count
                        )

                        prediction_text = f"{final_prediction} ({confidence:.2f})"
                    else:
                        prediction_text = ""

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(frame, prediction_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("AI Sign Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()