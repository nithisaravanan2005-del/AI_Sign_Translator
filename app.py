from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
import os
import time
from collections import deque, Counter

app = Flask(__name__)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("models/sign_model.h5")

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

reverse_label_map = {v: k for k, v in label_map.items()}

# ---------------- CUSTOM LOAD ----------------
def load_custom():
    try:
        with open("models/custom_signs.json", "r") as f:
            return json.load(f)
    except:
        return {}

def save_custom(data):
    with open("models/custom_signs.json", "w") as f:
        json.dump(data, f, indent=4)

custom_signs = load_custom()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- GLOBAL ----------------
prediction_buffer = deque(maxlen=5)
current_prediction = ""

training_mode = False
training_name = ""
training_data = []
training_start_time = 0

# ---------------- LANDMARK FIX ----------------
def get_landmarks(results):
    landmarks = []

    if results.multi_hand_landmarks:
        hands_list = results.multi_hand_landmarks

        # always 2-hand structure
        for i in range(2):
            if i < len(hands_list):
                for lm in hands_list[i].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0]*63)

    return landmarks


# ---------------- VIDEO ----------------
def generate_frames():
    global current_prediction, training_mode, training_data, training_start_time

    while True:
        success, frame = cap.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        temp_prediction = ""

        landmarks = get_landmarks(results)

        if len(landmarks) == 126:

            # ---------------- TRAIN ----------------
            if training_mode:
                training_data.append(landmarks.copy())

                remaining = int(5 - (time.time() - training_start_time))
                cv2.putText(frame, f"Training {training_name} ({remaining}s)",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                if time.time() - training_start_time > 5:
                    training_mode = False

                    if len(training_data) > 5:
                        if training_name not in custom_signs:
                            custom_signs[training_name] = []

                        custom_signs[training_name].extend(training_data)
                        save_custom(custom_signs)

                    training_data.clear()

            # ---------------- CUSTOM MATCH ----------------
            best_match = ""
            best_score = 999

            for name, samples in custom_signs.items():
                for sample in samples:
                    dist = np.linalg.norm(np.array(landmarks) - np.array(sample))

                    if dist < best_score:
                        best_score = dist
                        best_match = name

            if best_score < 0.5:
                temp_prediction = best_match.upper()

            # ---------------- MODEL ----------------
            else:
                input_data = np.array(landmarks).reshape(1, -1)

                pred = model.predict(input_data, verbose=0)
                class_index = np.argmax(pred)
                confidence = np.max(pred)

                if confidence > 0.6:
                    temp_prediction = reverse_label_map[class_index].upper()

        # DRAW
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # SMOOTH
        if temp_prediction != "":
            prediction_buffer.append(temp_prediction)

        if len(prediction_buffer) > 2:
            current_prediction = Counter(prediction_buffer).most_common(1)[0][0]

        # TEXT
        cv2.putText(frame, current_prediction, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({"prediction": current_prediction})

@app.route('/train_custom', methods=['POST'])
def train_custom():
    global training_mode, training_name, training_data, training_start_time

    data = request.json
    training_name = data.get("sign").upper()

    training_mode = True
    training_data = []
    training_start_time = time.time()

    return "started"

@app.route('/get_custom')
def get_custom():
    return jsonify(list(custom_signs.keys()))

@app.route('/delete_custom', methods=['POST'])
def delete_custom():
    global custom_signs
    data = request.json
    sign = data.get("sign")

    if sign in custom_signs:
        del custom_signs[sign]
        save_custom(custom_signs)

    return "deleted"


if __name__ == "__main__":
    app.run(debug=True)