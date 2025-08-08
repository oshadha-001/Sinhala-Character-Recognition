import tkinter as tk
from PIL import ImageTk, Image, ImageDraw
import cv2
import numpy as np
import os
import joblib

# Create data directory
try:
    os.mkdir('data')
except:
    print('Path Already Exists')

# Initialize Tkinter window
width, height = 500, 500
win = tk.Tk()
win.title("Sinhala Character Recognition")
font_btn = 'Helvetica 20 bold'
font_label = 'Helvetica 22 bold'
count = 0

# Load model and LabelEncoder
try:
    clsfr = joblib.load('sinhala-character-knn.sav')
    le = joblib.load('label_encoder.sav')
    print(f"Model loaded. Classes: {len(le.classes_)}")
except Exception as e:
    print(f"Error loading files: {e}")
    raise

def center_image(img_array):
    """Center the character in the image"""
    moments = cv2.moments(img_array)
    if moments['m00'] == 0:  # No non-zero pixels
        print("Warning: Empty image, skipping centering")
        return img_array
    cx = int(moments['mu10'] / moments['m00'])
    cy = int(moments['mu01'] / moments['m00'])
    shift_x = (img_array.shape[1] // 2) - cx
    shift_y = (img_array.shape[0] // 2) - cy
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img_array = cv2.warpAffine(img_array, M, (img_array.shape[1], img_array.shape[0]))
    return img_array

def event_function(event):
    print(f"Mouse event at: ({event.x}, {event.y})")  # Debug input
    x = event.x
    y = event.y
    x1, y1, x2, y2 = x - 10, y - 10, x + 10, y + 10  # Adjusted brush size
    canvas.create_oval((x1, y1, x2, y2), fill='black')
    img_draw.ellipse((x1, y1, x2, y2), fill='black')

def save():
    global count
    try:
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if np.all(img_array == 255):  # Check for blank canvas
            print("Save error: Blank canvas")
            label_status.config(text='ERROR: Please draw a character')
            return
        img_array = center_image(img_array)
        img_array = cv2.resize(img_array, (8, 8))
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        img_array = 255 - img_array  # Invert
        path = os.path.join('data', f'{count}.jpg')
        cv2.imwrite(path, img_array)
        count += 1
        print(f"Saved image: {path}")
        label_status.config(text=f'Saved image: {path}')
    except Exception as e:
        print(f"Save error: {e}")
        label_status.config(text=f'SAVE ERROR: {str(e)}')

def clear():
    global img, img_draw
    canvas.delete('all')
    img = Image.new('RGB', (width, height), (255, 255, 255))
    img_draw = ImageDraw.Draw(img)
    label_status.config(text='PREDICTED CHARACTER: NONE')
    print("Canvas cleared")

def predict():
    try:
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if np.all(img_array == 255):  # Check for blank canvas
            print("Predict error: Blank canvas")
            label_status.config(text='ERROR: Please draw a character')
            return
        img_array = center_image(img_array)
        img_array = cv2.resize(img_array, (8, 8))
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        img_array = 255 - img_array  # Invert
        img_array = img_array.reshape(1, -1) / 255.0  # Reshape and normalize
        result = clsfr.predict(img_array)[0]
        label = le.inverse_transform([result])[0]
        label_status.config(text=f'PREDICTED CHARACTER: {label}')
        print(f"Predicted: {label}")
    except Exception as e:
        print(f"Prediction error: {e}")
        label_status.config(text=f'ERROR: {str(e)}')

# Set up Tkinter widgets
canvas = tk.Canvas(win, width=width, height=height, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

button_save = tk.Button(win, text='SAVE', bg='green', fg='white', font=font_btn, command=save)
button_save.grid(row=1, column=0)

button_predict = tk.Button(win, text='PREDICT', bg='blue', fg='white', font=font_btn, command=predict)
button_predict.grid(row=1, column=1)

button_clear = tk.Button(win, text='CLEAR', bg='orange', fg='white', font=font_btn, command=clear)
button_clear.grid(row=1, column=2)

button_exit = tk.Button(win, text='EXIT', bg='red', fg='white', font=font_btn, command=win.destroy)
button_exit.grid(row=1, column=3)

label_status = tk.Label(win, text='PREDICTED CHARACTER: NONE', bg='white', font=font_label)
label_status.grid(row=2, column=0, columnspan=4)

# Bind mouse events
canvas.bind('<B1-Motion>', event_function)
canvas.bind('<Button-1>', event_function)
print("Mouse bindings set")

img = Image.new('RGB', (width, height), (255, 255, 255))
img_draw = ImageDraw.Draw(img)

win.mainloop()