import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
from collections import deque, Counter
import pyttsx3
import threading
import time

# ---------------- VOICE ----------------
engine = pyttsx3.init()
last_spoken = ""
last_spoken_time = 0

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- MODEL ----------------
model = tf.keras.models.load_model("models/sign_model.h5")

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

reverse_label_map = {v: k for k, v in label_map.items()}

# ---------------- CUSTOM ----------------
try:
    with open("models/custom_signs.json", "r") as f:
        custom_signs = json.load(f)
except:
    custom_signs = {}

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# ---------------- STABILITY ----------------
prediction_buffer = deque(maxlen=15)
stable_prediction = ""

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_prediction = ""

    if results.multi_hand_landmarks:

        landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) < 126:
            landmarks.extend([0] * (126 - len(landmarks)))

        # -------- CUSTOM MATCH --------
        matched = False
        for sign_name, saved_landmarks in custom_signs.items():
            dist = euclidean_distance(landmarks, saved_landmarks)

            if dist < 0.85:
                current_prediction = sign_name
                matched = True
                break

        # -------- AI MODEL --------
        if not matched:
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data, verbose=0)

            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            if confidence > 0.75:
                current_prediction = reverse_label_map[class_index]

    # -------- SMOOTHING --------
    if current_prediction != "":
        prediction_buffer.append(current_prediction)

    if len(prediction_buffer) > 5:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]

        if most_common != stable_prediction:
            stable_prediction = most_common

    display_text = stable_prediction

    # -------- DRAW --------
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # -------- VOICE --------
    current_time = time.time()

    if display_text != "":
        if display_text != last_spoken or (current_time - last_spoken_time > 2):
            threading.Thread(target=speak, args=(display_text,)).start()
            last_spoken = display_text
            last_spoken_time = current_time

    # -------- DISPLAY --------
    cv2.putText(frame, display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3)

    cv2.imshow("AI Sign Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()