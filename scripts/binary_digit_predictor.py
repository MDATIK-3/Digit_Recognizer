import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
import os

def image_to_binary_string(img: Image.Image) -> str:
    img = img.convert("L")  
    arr = np.array(img)
    return ''.join(['1' if pixel < 128 else '0' for pixel in arr.flatten()])

def load_binary_dataset(language="en"):
    """Load a binary dataset based on language"""
    file = f"binary_dataset_{language}.txt"
    
    if not os.path.exists(file):
        print(f"⚠️ Warning: Dataset file {file} not found. Checking for default dataset file.")
        file = "binary_dataset.txt"
        if not os.path.exists(file):
            print(f"❌ Error: No dataset file found")
            return []
    
    dataset = []
    try:
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    binary, label = parts
                    dataset.append((binary, int(label)))
                else:
                    print(f"Skipping invalid line: {line}")
        
        print(f"✅ Loaded {len(dataset)} samples from {file}")
        return dataset
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

def hamming_distance(s1, s2):
    """Calculate Hamming distance between two binary strings of equal length"""
    # Handle case where strings might be of different lengths
    min_len = min(len(s1), len(s2))
    return sum(a != b for a, b in zip(s1[:min_len], s2[:min_len]))

def predict_digit_from_binary(binary_input, dataset):
    """Predict digit using Hamming distance"""
    if not dataset:
        return None, 0
        
    distances = [(x[1], hamming_distance(x[0], binary_input)) for x in dataset]
    closest_label, min_distance = min(distances, key=lambda x: x[1])
    max_distance = len(binary_input)
    confidence = 1 - (min_distance / max_distance)
    return closest_label, confidence

class MultilingualDigitPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multilingual Digit Recognizer")
        
        # Datasets for different languages
        self.datasets = {
            "en": {"name": "English", "data": []},
            "bn": {"name": "Bangla", "data": []}
        }
        
        # Available languages
        self.available_languages = []
        for lang in self.datasets.keys():
            dataset = load_binary_dataset(lang)
            if dataset:
                self.datasets[lang]["data"] = dataset
                self.available_languages.append(lang)
        
        if not self.available_languages:
            print("No language datasets found. Trying default dataset...")
            default_dataset = load_binary_dataset()
            if default_dataset:
                self.datasets["en"]["data"] = default_dataset
                self.available_languages.append("en")
        
        if not self.available_languages:
            tk.messagebox.showerror("Error", "No datasets found. Please generate datasets first.")
            root.destroy()
            return
            
        print(f"Available languages: {', '.join(lang for lang in self.available_languages)}")
        
        self.current_language = self.available_languages[0]
        
        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # Drawing area
        self.canvas_width = 250
        self.canvas_height = 300
        
        drawing_frame = tk.LabelFrame(main_frame, text="Draw Digit")
        drawing_frame.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(
            drawing_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="white", 
            highlightthickness=1, 
            highlightbackground="black"
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame)
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        
        # Language selection
        lang_frame = tk.LabelFrame(control_frame, text="Select Language")
        lang_frame.pack(fill="x", padx=5, pady=5)
        
        self.language_var = tk.StringVar(value=self.current_language)
        
        for i, lang_code in enumerate(self.available_languages):
            lang_name = self.datasets[lang_code]["name"]
            tk.Radiobutton(
                lang_frame,
                text=lang_name,
                variable=self.language_var,
                value=lang_code,
                command=self.change_language
            ).pack(anchor="w", padx=10, pady=2)
        
        # Action buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        predict_btn = tk.Button(
            button_frame, 
            text="Predict", 
            command=self.predict_digit, 
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            width=10,
            height=2
        )
        predict_btn.pack(pady=5)
        
        clear_btn = tk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear_canvas, 
            font=("Arial", 10, "bold"),
            width=10
        )
        clear_btn.pack(pady=5)
        
        # Results area
        results_frame = tk.LabelFrame(control_frame, text="Prediction Result")
        results_frame.pack(fill="x", padx=5, pady=5)
        
        self.result_label = tk.Label(
            results_frame, 
            text="Draw a digit and click Predict", 
            font=("Arial", 10),
            wraplength=200,
            justify="center",
            pady=10
        )
        self.result_label.pack(fill="x")
        
        self.confidence_meter = ttk.Progressbar(
            results_frame, 
            orient="horizontal", 
            length=200, 
            mode="determinate"
        )
        self.confidence_meter.pack(padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.update_status()
        status_bar = tk.Label(
            root, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Setup drawing canvas
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        
        self.prev_x = None
        self.prev_y = None
        
    def start_draw(self, event):
        self.prev_x = event.x
        self.prev_y = event.y
        
    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        
        # Draw line for smoother drawing
        if self.prev_x and self.prev_y:
            self.canvas.create_line(
                self.prev_x, self.prev_y, x, y,
                width=r*2, 
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE
            )
            self.draw_obj.line(
                [self.prev_x, self.prev_y, x, y],
                fill=0,
                width=r*2
            )
        else:
            # Just a point
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
            self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill=0)
            
        self.prev_x = x
        self.prev_y = y
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None
        self.result_label.config(
            text="Draw a digit and click Predict",
            fg="black",
            font=("Arial", 10)
        )
        self.confidence_meter["value"] = 0
        
    def change_language(self):
        self.current_language = self.language_var.get()
        self.update_status()
        self.clear_canvas()
        
    def update_status(self):
        lang_name = self.datasets[self.current_language]["name"]
        dataset_size = len(self.datasets[self.current_language]["data"])
        self.status_var.set(f"Mode: {lang_name} | Dataset Size: {dataset_size} samples")
        
    def predict_digit(self):
        binary = image_to_binary_string(self.image)
        dataset = self.datasets[self.current_language]["data"]
        
        if not dataset:
            self.result_label.config(
                text=f"No dataset available for {self.datasets[self.current_language]['name']}",
                fg="red"
            )
            return
            
        digit, confidence = predict_digit_from_binary(binary, dataset)
        
        if digit is None:
            self.result_label.config(
                text="Could not make a prediction",
                fg="red"
            )
            return
            
        confidence_percent = round(confidence * 100, 2)
        self.confidence_meter["value"] = confidence_percent
        
        # Adjust text color based on confidence
        color = "green" if confidence_percent > 70 else "orange" if confidence_percent > 50 else "red"
        
        self.result_label.config(
            text=f"Predicted Digit: {digit}\nConfidence: {confidence_percent}%",
            fg=color,
            font=("Arial", 12, "bold")
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = MultilingualDigitPredictorApp(root)
    root.mainloop()