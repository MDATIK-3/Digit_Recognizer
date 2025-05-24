import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk

def image_to_binary_string(img: Image.Image) -> str:
    img = img.convert("L")  
    arr = np.array(img)
    return ''.join(['1' if pixel < 255 else '0' for pixel in arr.flatten()])

def load_binary_dataset(file="binary_dataset_bn.txt"):
    dataset = []
    with open(file, "r") as f:
        for line in f:
            binary, label = line.strip().split(",")
            dataset.append((binary, int(label)))
    return dataset

def hamming_distance(s1, s2):
    return sum(a != b for a, b in zip(s1, s2))

def predict_digit_from_binary(binary_input, dataset):
    closest = min(dataset, key=lambda x: hamming_distance(x[0], binary_input))
    return closest[1]

class BinaryDigitPredictorApp:
    def __init__(self):
        self.dataset = load_binary_dataset()

        self.window = tk.Tk()
        self.window.title("Binary Digit Recognizer")

        self.canvas_width = 250
        self.canvas_height = 300

        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.pack(pady=10)

        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Predict", command=self.predict_digit, font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=5)
        tk.Button(self.button_frame, text="Clear", command=self.clear_canvas, font=("Arial", 10, "bold")).grid(row=0, column=1, padx=10, pady=5)

        self.result_label = tk.Label(self.window, text="Draw a digit and click Predict", font=("Arial", 10))
        self.result_label.pack(pady=5)

        self.canvas.bind("<B1-Motion>", self.draw)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_obj.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.result_label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        binary = image_to_binary_string(self.image)
        digit = predict_digit_from_binary(binary, self.dataset)
        self.result_label.config(text=f"Predicted Digit: {digit}", fg="green", font=("Arial", 12, "bold"))

    def run(self):
        self.window.mainloop()

app = BinaryDigitPredictorApp()
app.run()
