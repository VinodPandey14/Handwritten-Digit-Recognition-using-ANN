import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tkinter import Tk, Canvas, Button
from PIL import ImageGrab, ImageOps
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('trained_model.h5')

# Preprocessing function to resize and normalize the canvas image
def preprocess_image(img):
    img = ImageOps.grayscale(img)  # Convert to grayscale
    img = img.resize((28, 28))    # Resize to 28x28
    img_array = np.array(img)     # Convert to NumPy array
    img_array = 255 - img_array   # Invert colors
    img_array = img_array / 255.0 # Normalize pixel values to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1)
    return img_array

# Predict the digit
def predict_digit():
    global canvas
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    w = x + canvas.winfo_width()
    h = y + canvas.winfo_height()

    # Grab the canvas content as an image
    img = ImageGrab.grab(bbox=(x, y, w, h))

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Predict using the model
    prediction = model.predict(preprocessed_img)
    predicted_digit = np.argmax(prediction)

    # Display the result
    print(f"Predicted Digit: {predicted_digit}")
    plt.imshow(preprocessed_img[0].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.axis('off')
    plt.show()

# Clear the canvas
def clear_canvas():
    global canvas
    canvas.delete("all")

# Main application
def main():
    global canvas
    root = Tk()
    root.title("Draw a Digit")

    # Create a canvas for drawing
    canvas = Canvas(root, width=300, height=300, bg="white")
    canvas.grid(row=0, column=0, columnspan=2)

    # Enable drawing on the canvas
    def paint(event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    canvas.bind("<B1-Motion>", paint)

    # Add buttons for prediction and clearing the canvas
    predict_button = Button(root, text="Predict", command=predict_digit)
    predict_button.grid(row=1, column=0)

    clear_button = Button(root, text="Clear", command=clear_canvas)
    clear_button.grid(row=1, column=1)

    root.mainloop()

if __name__ == "__main__":
    main()
