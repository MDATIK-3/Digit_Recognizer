import tkinter as tk
from PIL import Image, ImageDraw
import os

class DigitDataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Data Collector")
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        self.canvas = tk.Canvas(main_frame, width=250, height=300, bg="white", bd=2, relief=tk.SUNKEN)
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)
        
        label_frame = tk.LabelFrame(control_frame, text="Select Digit")
        label_frame.pack(pady=10, fill=tk.X)
        
        self.label = tk.StringVar(value="0")
        self.create_label_buttons(label_frame)
        
        save_frame = tk.LabelFrame(control_frame, text="Save Options")
        save_frame.pack(pady=10, fill=tk.X)
        
        self.save_button = tk.Button(save_frame, text="Save Now", command=self.save_image)
        self.save_button.pack(pady=5, fill=tk.X)
        
        auto_save_frame = tk.Frame(save_frame)
        auto_save_frame.pack(pady=5, fill=tk.X)
        
        self.auto_save = tk.BooleanVar(value=False)
        self.auto_save_check = tk.Checkbutton(
            auto_save_frame, text="Auto Save", variable=self.auto_save,
            command=self.toggle_auto_save
        )
        self.auto_save_check.pack(side=tk.LEFT)
        
        tk.Label(save_frame, text="Auto-save interval (seconds):").pack(anchor=tk.W)
        
        self.timer_interval = tk.IntVar(value=5)
        timer_values = [5, 8, 10, 30]
        timer_frame = tk.Frame(save_frame)
        timer_frame.pack(fill=tk.X)
        
        for val in timer_values:
            tk.Radiobutton(
                timer_frame, text=str(val), variable=self.timer_interval, 
                value=val, command=self.update_timer
            ).pack(side=tk.LEFT)
        
        self.timer_active = False
        self.timer_id = None
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(
            save_frame, textvariable=self.status_var, fg="blue", 
            relief=tk.SUNKEN, bd=1, padx=5, pady=3
        )
        self.status_label.pack(pady=5, fill=tk.X)
        
        self.clear_button = tk.Button(
            control_frame, text="Clear Canvas", command=self.clear_canvas
        )
        self.clear_button.pack(pady=10, fill=tk.X)
        
        counter_frame = tk.LabelFrame(control_frame, text="Image Count")
        counter_frame.pack(pady=10, fill=tk.X)
        
        self.counter_var = tk.StringVar(value="Total: 0")
        self.counter_label = tk.Label(counter_frame, textvariable=self.counter_var)
        self.counter_label.pack(pady=5)
        
        self.previous_x = None
        self.previous_y = None
        self.image = Image.new("L", (250, 300), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.has_content = False
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        self.update_counter()
        
    def create_label_buttons(self, parent):
        digits_frame = tk.Frame(parent)
        digits_frame.pack(pady=5)
        
        for i in range(10):
            col = i % 5
            row = i // 5
            button = tk.Radiobutton(
                digits_frame, text=str(i), variable=self.label, 
                value=str(i), width=3, indicatoron=0,
                command=self.update_counter
            )
            button.grid(row=row, column=col, padx=2, pady=2, sticky=tk.W+tk.E)
    
    def paint(self, event):
        x, y = event.x, event.y
        if self.previous_x and self.previous_y:
            self.canvas.create_line(
                (self.previous_x, self.previous_y, x, y),
                width=8,
                fill="black",
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.draw.line(
                [self.previous_x, self.previous_y, x, y], fill="black", width=8
            )
            self.has_content = True
        self.previous_x = x
        self.previous_y = y
    
    def reset(self, event):
        self.previous_x = None
        self.previous_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (250, 300), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.has_content = False
    
    def save_image(self):
        if not self.has_content:
            self.show_status("Nothing to save", "red")
            return
            
        label = self.label.get()
        folder_path = f"dataset/{label}"
        os.makedirs(folder_path, exist_ok=True)
        
        existing_files = os.listdir(folder_path)
        numbers = []
        
        for f in existing_files:
            name, ext = os.path.splitext(f)
            if ext.lower() == ".png" and name.isdigit():
                numbers.append(int(name))
        
        next_num = max(numbers, default=0) + 1
        filename = os.path.join(folder_path, f"{next_num}.png")
        
        img_to_save = self.image.copy()
        img_to_save = img_to_save.convert("L")  
        img_to_save.save(filename)
        
        self.show_status(f"Saved to {filename}", "green")
        self.clear_canvas()
        self.update_counter()
    
    def auto_save_timer(self):
        if self.timer_active and self.auto_save.get():
            if self.has_content:
                self.save_image()
                self.show_status(f"Auto-saved (every {self.timer_interval.get()}s)", "blue")
            self.timer_id = self.root.after(self.timer_interval.get() * 1000, self.auto_save_timer)
    
    def toggle_auto_save(self):
        self.timer_active = self.auto_save.get()
        if self.timer_active:
            self.show_status(f"Auto-save active ({self.timer_interval.get()}s)", "blue")
            self.auto_save_timer()
        else:
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
                self.timer_id = None
            self.show_status("Auto-save disabled", "blue")
    
    def update_timer(self):
        if self.timer_active:
            if self.timer_id:
                self.root.after_cancel(self.timer_id)
            self.show_status(f"Auto-save interval: {self.timer_interval.get()}s", "blue")
            self.timer_id = self.root.after(self.timer_interval.get() * 1000, self.auto_save_timer)
    
    def show_status(self, message, color="black"):
        self.status_var.set(message)
        self.status_label.config(fg=color)
        self.root.after(3000, lambda: self.status_var.set("Ready") or self.status_label.config(fg="blue"))
    
    def update_counter(self):
        label = self.label.get()
        folder_path = f"dataset/{label}"
        
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith('.png')])
        else:
            count = 0
            
        self.counter_var.set(f"Digit {label}: {count} images")

root = tk.Tk()
app = DigitDataCollectorApp(root)
root.mainloop()