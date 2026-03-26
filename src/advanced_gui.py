import tkinter as tk
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import json
from collections import deque, Counter

# ---------------- LOAD MODEL ----------------
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

# ---------------- MAIN WINDOW ----------------
root = tk.Tk()
root.title("AI Sign Translator")
root.geometry("700x600")

label = tk.Label(root)
label.pack()

text_label = tk.Label(root, text="", font=("Arial", 20))
text_label.pack(pady=10)

running = True

# ---------------- UPDATE FUNCTION ----------------
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

    # Show text
    text_label.config(text=stable_prediction)

    # Convert to GUI image
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

    root.after(10, update_frame)

# ---------------- BUTTONS ----------------
def start():
    global running
    running = True
    update_frame()

def stop():
    global running
    running = False

btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Start", command=start, width=15).pack(side="left", padx=10)
tk.Button(btn_frame, text="Stop", command=stop, width=15).pack(side="left", padx=10)
tk.Button(btn_frame, text="Exit", command=root.quit, width=15).pack(side="left", padx=10)

# Start automatically
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()