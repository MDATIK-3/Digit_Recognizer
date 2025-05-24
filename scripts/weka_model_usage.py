import os
import sys
import numpy as np
from PIL import Image
import subprocess
import tempfile
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk

class WekaModelPredictorBinary:
    def __init__(self, image_width=28, image_height=28):
        # Use the hardcoded paths instead of searching
        self.weka_jar_path = "C:\\Program Files\\Weka-3-8-6\\weka.jar"
        self.model_path = "digit_recognition_model.model"
        self.arff_path = "D:\\GUB\\AI_Lab\\Digit_Recognizer-main\\Digit_Recognizer_final\\digit_images.arff"
        
        # For J28 format, make sure we use exactly 28x28 pixels
        self.image_width = 28
        self.image_height = 28
        
        # Check if Weka JAR exists
        if not os.path.exists(self.weka_jar_path):
            print(f"Warning: Weka JAR not found at: {self.weka_jar_path}")
            # Try to use the JAR in the current directory as fallback
            if os.path.exists("weka.jar"):
                self.weka_jar_path = "weka.jar"
                print(f"Using fallback Weka JAR: {self.weka_jar_path}")
            else:
                raise ValueError(f"Weka JAR not found at: {self.weka_jar_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at: {self.model_path}")
            # Don't try to find the model file - just report error
            raise ValueError(f"Model file not found at: {self.model_path}")
            
        print(f"Using Weka JAR: {self.weka_jar_path}")
        print(f"Using model: {self.model_path}")
        print(f"Using ARFF reference: {self.arff_path}")
        print(f"Using image format: {self.image_width}x{self.image_height} pixels (784 binary features total)")
    
    def preprocess_image(self, image_path):
        """
        Preprocess image to match the training data format:
        1. Convert to grayscale
        2. Resize to 28x28 pixels
        3. Convert to binary values (0 or 1) based on threshold 127
        4. Flatten to 1D array of exactly 784 values
        """
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((self.image_width, self.image_height))  # Resize to 28x28
        
        # Convert to numpy array
        pixels = np.array(img)
        
        # Apply binary threshold (same as training data preprocessing)
        # Note: For digit recognition, typically dark pixels (low values) are foreground (1)
        # and light pixels (high values) are background (0)
        binary_pixels = (pixels <= 127).astype(int).flatten()
        
        # Ensure we have exactly 784 pixels (28x28)
        if len(binary_pixels) != 784:
            print(f"Warning: Expected 784 pixels, got {len(binary_pixels)}")
            if len(binary_pixels) < 784:
                binary_pixels = np.pad(binary_pixels, (0, 784 - len(binary_pixels)), 'constant')
            else:
                binary_pixels = binary_pixels[:784]
        
        print(f"Image preprocessed: {self.image_width}x{self.image_height} -> {len(binary_pixels)} binary features")
        print(f"Binary pixel distribution: {np.sum(binary_pixels)} ones, {len(binary_pixels) - np.sum(binary_pixels)} zeros")
        
        return binary_pixels
    
    def create_temp_arff(self, pixel_values):
        """Create ARFF file with binary pixel values to match training format"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.arff', delete=False)
        temp_path = temp_file.name
        
        # Ensure we have exactly 784 pixel values (28x28)
        num_attributes = len(pixel_values)
        if num_attributes != 784:
            print(f"Warning: Expected 784 pixels for 28x28 image, got {num_attributes}")
            # Pad or truncate if necessary
            if num_attributes < 784:
                pixel_values = np.pad(pixel_values, (0, 784 - num_attributes), 'constant')
            else:
                pixel_values = pixel_values[:784]
        
        arff_content = "@RELATION digit_images\n\n"
        
        # Define attributes as binary (0,1) to match training data
        for i in range(784):  # Always use 784 pixels for 28x28 image
            arff_content += f"@ATTRIBUTE pixel{i} {{0,1}}\n"
        
        # Define class attribute with possible class labels
        arff_content += "@ATTRIBUTE class {bn_0,bn_1,bn_2,bn_3,bn_4,bn_5,bn_6,bn_7,bn_8,bn_9,en_0,en_1,en_2,en_3,en_4,en_5,en_6,en_7,en_8,en_9}\n\n"
        arff_content += "@DATA\n"
        
        # Convert pixel values to binary strings
        pixel_str = ",".join([str(int(val)) for val in pixel_values])
        arff_content += f"{pixel_str},?\n"
        
        temp_file.write(arff_content.encode('utf-8'))
        temp_file.close()
        
        print(f"Created temporary ARFF file: {temp_path}")
        return temp_path
    
    def predict(self, image_path):
        pixel_values = self.preprocess_image(image_path)
        temp_arff = self.create_temp_arff(pixel_values)
        
        try:
            # Primary command for prediction
            cmd = [
                "java", "-Xmx4g", "-cp", self.weka_jar_path,
                "weka.classifiers.trees.RandomForest",
                "-T", temp_arff,
                "-l", self.model_path,
                "-p", "0"
            ]
            
            print(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print(f"Return code: {result.returncode}")
            print(f"Stdout: {result.stdout}")
            if result.stderr:
                print(f"Stderr: {result.stderr}")
            
            # Try alternative command if first fails
            weka_error = False
            if result.returncode != 0:
                weka_error = True
                cmd_alt = [
                    "java", "-Xmx4g", "-cp", self.weka_jar_path,
                    "weka.classifiers.misc.InputMappedClassifier",
                    "-L", self.model_path,
                    "-T", temp_arff,
                    "-p", "0"
                ]
                
                print(f"Trying alternative command: {' '.join(cmd_alt)}")
                result = subprocess.run(cmd_alt, capture_output=True, text=True)
                
                if result.returncode == 0:
                    weka_error = False
                    
            # If all attempts fail, try different format
            if weka_error or not result.stdout.strip():
                # Try with J48 classifier as fallback
                cmd_j48 = [
                    "java", "-Xmx4g", "-cp", self.weka_jar_path,
                    "weka.classifiers.trees.J48",
                    "-T", temp_arff,
                    "-l", self.model_path,
                    "-p", "0"
                ]
                
                print(f"Trying J48 classifier command: {' '.join(cmd_j48)}")
                result = subprocess.run(cmd_j48, capture_output=True, text=True)
                    
                if result.returncode != 0:
                    raise RuntimeError(f"All Weka prediction attempts failed: {result.stderr}")
            
            output = result.stdout
            print(f"Raw Weka output:\n{output}")
            
            # Parse the prediction output
            predicted_class, confidence, all_confidences = self.parse_weka_output(output)
            
            if predicted_class is None:
                raise ValueError(f"Could not parse Weka output. Raw output:\n{output}")
            
            return predicted_class, confidence, all_confidences
                
        finally:
            if os.path.exists(temp_arff):
                os.remove(temp_arff)
    
    def parse_weka_output(self, output):
        """Parse Weka output to extract prediction and confidence"""
        predicted_class = None
        confidence = 0.0
        all_confidences = {}
        
        # Multiple patterns to handle different Weka output formats
        patterns = [
            # Pattern for detailed output with confidence distribution
            r'1\s+\?\s+([a-z]+_\d+)\s+([\d\.]+)\s+\[([\d\.,\s]+)\]',
            # Pattern for simple output
            r'1\s+\?\s+([a-z]+_\d+)\s+([\d\.]+)',
            # Alternative patterns
            r'1,\?,([a-z]+_\d+),([\d\.]+),\[([\d\.,\s]+)\]',
            r'1:[\s]*\?[\s]*([a-z]+_\d+)[\s]*([\d\.]+)[\s]*\[([\d\.,\s]+)\]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output, re.MULTILINE)
            if match:
                predicted_class = match.group(1)
                confidence = float(match.group(2))
                
                # Extract confidence distribution if available
                if len(match.groups()) > 2:
                    try:
                        conf_str = match.group(3).strip('[]')
                        conf_values = [float(x.strip()) for x in conf_str.split(',') if x.strip()]
                        
                        # Map confidence values to class labels
                        # This is a simplified mapping - you might need to adjust based on your model
                        class_labels = ['bn_0','bn_1','bn_2','bn_3','bn_4','bn_5','bn_6','bn_7','bn_8','bn_9',
                                      'en_0','en_1','en_2','en_3','en_4','en_5','en_6','en_7','en_8','en_9']
                        
                        for i, conf_val in enumerate(conf_values):
                            if i < len(class_labels):
                                all_confidences[class_labels[i]] = conf_val
                                
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing confidence distribution: {e}")
                        all_confidences[predicted_class] = confidence
                else:
                    all_confidences[predicted_class] = confidence
                
                break
        
        # If no pattern matched, try simple line-by-line parsing
        if predicted_class is None:
            lines = output.strip().split('\n')
            for line in lines:
                if re.search(r'^\s*1[\s,]', line):
                    parts = re.split(r'[\s,]+', line.strip())
                    for part in parts:
                        if '_' in part and any(char.isdigit() for char in part):
                            predicted_class = part
                            confidence = 1.0
                            all_confidences[predicted_class] = confidence
                            break
                    if predicted_class:
                        break
        
        # Extract just the digit and language from the class label
        if predicted_class:
            try:
                # Parse class label like "en_5" or "bn_3"
                parts = predicted_class.split('_')
                if len(parts) == 2:
                    language, digit = parts
                    digit = int(digit)
                    
                    # Create simplified confidences for display (just the digit)
                    simplified_confidences = [0.0] * 10
                    if predicted_class in all_confidences:
                        simplified_confidences[digit] = all_confidences[predicted_class]
                    else:
                        simplified_confidences[digit] = confidence
                    
                    return f"{language}_{digit}", confidence, simplified_confidences
            except (ValueError, IndexError):
                pass
        
        return predicted_class, confidence, [0.0] * 10


class DigitPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Binary Digit Predictor (Weka Model - j28 Format)")
        
        self.predictor = None
        self.model_loaded = False
        
        # Use larger canvas for better drawing, but will be resized to 28x28 for prediction
        self.canvas_width = 280
        self.canvas_height = 280
        
        self.create_widgets()
        
        self.prev_x = None
        self.prev_y = None
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.has_content = False
        
        try:
            # Always use 28x28 for j28 format
            self.predictor = WekaModelPredictorBinary(image_width=28, image_height=28)
            self.model_loaded = True
            self.model_info_var.set(f"Model: Binary Digit Classifier (j28)\nStatus: Loaded\nFormat: 28x28 binary pixels (784 features)")
            self.predict_btn['state'] = 'normal'
            self.status_var.set("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_info_var.set("No model loaded")
            self.predict_btn['state'] = 'disabled'
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Drawing frame
        drawing_frame = ttk.LabelFrame(main_frame, text="Draw Digit (will be resized to 28x28)")
        drawing_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.canvas = tk.Canvas(
            drawing_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="white", 
            highlightthickness=1, 
            highlightbackground="black"
        )
        self.canvas.pack(padx=10, pady=10)
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Model info frame
        model_frame = ttk.LabelFrame(control_frame, text="Model Information")
        model_frame.pack(fill="x", padx=5, pady=5)
        
        self.model_info_var = tk.StringVar(value="Loading model...")
        ttk.Label(
            model_frame, 
            textvariable=self.model_info_var, 
            wraplength=200
        ).pack(fill="x", padx=5, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        self.predict_btn = ttk.Button(
            button_frame, 
            text="Predict", 
            command=self.predict_digit,
            state="disabled"
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear_canvas
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Load Image",
            command=self.load_image
        ).pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(control_frame, text="Prediction Result")
        results_frame.pack(fill="x", padx=5, pady=5)
        
        self.result_label = ttk.Label(
            results_frame, 
            text="Draw a digit and click Predict", 
            font=("Arial", 12),
            wraplength=200,
            justify="center"
        )
        self.result_label.pack(fill="x", padx=5, pady=10)
        
        self.details_label = ttk.Label(
            results_frame, 
            text="", 
            font=("Arial", 9),
            wraplength=200,
            justify="center"
        )
        self.details_label.pack(fill="x", padx=5, pady=5)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        
        if not image_path:
            return
            
        try:
            img = Image.open(image_path).convert('L')
            self.clear_canvas()
            
            # Resize to canvas size for display
            img_display = img.resize((self.canvas_width, self.canvas_height), Image.LANCZOS)
            
            # Store the original size image for prediction
            self.image = img_display
            self.draw_obj = ImageDraw.Draw(self.image)
            
            # Display on canvas
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
            self.has_content = True
            self.status_var.set(f"Loaded image: {os.path.basename(image_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def start_draw(self, event):
        self.prev_x = event.x
        self.prev_y = event.y
        
    def draw(self, event):
        x, y = event.x, event.y
        r = 10  # Brush radius
        
        if self.prev_x and self.prev_y:
            # Draw line on canvas
            self.canvas.create_line(
                self.prev_x, self.prev_y, x, y,
                width=r*2, 
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE
            )
            # Draw line on PIL image
            self.draw_obj.line(
                [self.prev_x, self.prev_y, x, y],
                fill=0,
                width=r*2
            )
        else:
            # Draw circle on canvas
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
            # Draw circle on PIL image
            self.draw_obj.ellipse([x-r, y-r, x+r, y+r], fill=0)
            
        self.prev_x = x
        self.prev_y = y
        self.has_content = True
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None
        self.has_content = False
        self.result_label.config(
            text="Draw a digit and click Predict",
            foreground="black"
        )
        self.details_label.config(text="")
            
    def predict_digit(self):
        if not self.model_loaded or not self.has_content:
            if not self.model_loaded:
                messagebox.showerror("Error", "No model loaded")
            elif not self.has_content:
                messagebox.showerror("Error", "Draw a digit first")
            return
            
        try:
            # Save current drawing to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            self.image.save(temp_path)
            
            self.status_var.set("Predicting...")
            self.root.update()
            
            # Make prediction
            predicted_class, confidence, all_confidences = self.predictor.predict(temp_path)
            
            # Parse the prediction
            if predicted_class and '_' in predicted_class:
                language, digit = predicted_class.split('_')
               
                
                confidence_percent = round(confidence * 100, 2)
                
                # Determine color based on confidence
                if confidence_percent > 70:
                    color = "green"
                elif confidence_percent > 50:
                    color = "orange"
                else:
                    color = "red"
                
                self.result_label.config(
                    text=f"Predicted: {digit}\nConfidence: {confidence_percent}%",
                    foreground=color
                )
                
                # Show additional details
                self.details_label.config(
                    text=f"Full class: {predicted_class}\nModel processed: 28x28 binary image"
                )
            else:
                self.result_label.config(
                    text=f"Predicted: {predicted_class}\nConfidence: {round(confidence * 100, 2)}%",
                    foreground="blue"
                )
                self.details_label.config(text="")
            
            self.status_var.set("Prediction complete")
            os.remove(temp_path)
            
        except Exception as e:
            error_msg = str(e)
            if "Could not parse Weka output" in error_msg:
                error_msg = ("Failed to parse Weka output. This might be due to:\n"
                           "1. Model expects different input format\n"
                           "2. Weka version compatibility issues\n"
                           "3. Model file corruption\n\n"
                           f"Technical details: {str(e)}")
            messagebox.showerror("Prediction Error", error_msg)
            self.status_var.set("Prediction failed")
            
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)


def main():
    root = tk.Tk()
    app = DigitPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()