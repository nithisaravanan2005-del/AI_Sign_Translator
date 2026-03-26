import tkinter as tk
import os

def collect_data():
    os.system("python src/data_collection.py")

def train_model():
    os.system("python src/preprocess.py")
    os.system("python src/train_model.py")

def run_prediction():
    os.system("python src/predict.py")

def custom_sign():
    os.system("python src/custom_sign.py")

root = tk.Tk()
root.title("AI Sign Translator")
root.geometry("400x400")

title = tk.Label(root, text="AI Sign Translator", font=("Arial", 16))
title.pack(pady=20)

tk.Button(root, text="Collect Data", width=25, command=collect_data).pack(pady=10)
tk.Button(root, text="Train Model", width=25, command=train_model).pack(pady=10)
tk.Button(root, text="Run Prediction", width=25, command=run_prediction).pack(pady=10)
tk.Button(root, text="Add Custom Sign", width=25, command=custom_sign).pack(pady=10)
tk.Button(root, text="Exit", width=25, command=root.quit).pack(pady=20)

root.mainloop()