import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
from collections import deque, Counter

# ---------------- MODEL ----------------
model = tf.keras.models.load_model("models/sign_model.h5")

with open("models/label_map.json", "r") as f:
    label_map = json.load(f)

reverse_label_map = {v: k for k, v in label_map.items()}

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=15)
stable_prediction = ""

running = False

# ---------------- MAIN WINDOW ----------------
root = tk.Tk()
root.title("AI Sign Translator")
root.geometry("900x600")
root.configure(bg="#1e1e1e")

# ---------------- SIDEBAR ----------------
sidebar = tk.Frame(root, bg="#2c2c2c", width=200)
sidebar.pack(side="left", fill="y")

# ---------------- MAIN AREA ----------------
main_area = tk.Frame(root, bg="#1e1e1e")
main_area.pack(side="right", fill="both", expand=True)

# Camera display
camera_label = tk.Label(main_area, bg="black")
camera_label.pack(pady=10)

# Prediction text
text_label = tk.Label(main_area, text="",
                      font=("Arial", 22),
                      fg="white",
                      bg="#1e1e1e")
text_label.pack(pady=10)

# ---------------- FUNCTIONS ----------------
def start():
    global running
    running = True
    update_frame()

def stop():
    global running
    running = False

def update_frame():
    global stable_prediction

    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

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

        input_data = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(input_data, verbose=0)

        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.75:
            current_prediction = reverse_label_map[class_index]

    if current_prediction != "":
        prediction_buffer.append(current_prediction)

    if len(prediction_buffer) > 5:
        stable_prediction = Counter(prediction_buffer).most_common(1)[0][0]

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    text_label.config(text=stable_prediction)

    # Convert to image
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    root.after(10, update_frame)

# ---------------- SIDEBAR BUTTONS ----------------
def create_button(text, command):
    return tk.Button(sidebar,
                     text=text,
                     command=command,
                     font=("Arial", 12),
                     bg="#3c3c3c",
                     fg="white",
                     bd=0,
                     pady=10)

create_button("Start", start).pack(fill="x", pady=5)
create_button("Stop", stop).pack(fill="x", pady=5)
create_button("Exit", root.quit).pack(fill="x", pady=5)

# ---------------- RUN ----------------
root.mainloop()

cap.release()
cv2.destroyAllWindows()