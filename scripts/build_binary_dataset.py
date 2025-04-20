import os
from PIL import Image

def image_to_binary_string(img):
    img = img.convert("L") 
    return ''.join(['1' if pixel < 255 else '0' for pixel in img.getdata()])

def build_binary_dataset(data_dir="dataset", output_file="binary_dataset.txt"):
    with open(output_file, "w") as f:
        for label in range(10):
            folder = os.path.join(data_dir, str(label))
            if not os.path.isdir(folder):
                print(f"Skipping missing folder: {folder}")
                continue
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(folder, file)
                    try:
                        img = Image.open(img_path)
                        binary = image_to_binary_string(img)
                        f.write(f"{binary},{label}\n")
                    except Exception as e:
                        print(f"❌ Error processing image {img_path}: {e}")
    print("✅ binary_dataset.txt generated successfully!")

if __name__ == "__main__":
    build_binary_dataset()
