import os
from PIL import Image
import argparse

def image_to_binary_string(img):
    img = img.convert("L") 
    return ''.join(['1' if pixel < 128 else '0' for pixel in img.getdata()])

def build_binary_dataset(data_dir="dataset", language="en", output_file=None):
    """
    Convert images to binary strings and save to a text file.
    
    Args:
        data_dir: Base directory containing the dataset
        language: Language code ('en' for English, 'bn' for Bangla)
        output_file: Output filename (default: binary_dataset_{language}.txt)
    """
    if output_file is None:
        output_file = f"binary_dataset_{language}.txt"
    
    language_dir = os.path.join(data_dir, language)
    if not os.path.isdir(language_dir):
        print(f"❌ Language directory not found: {language_dir}")
        return
    
    count = 0
    with open(output_file, "w") as f:
        for label in range(10):
            folder = os.path.join(language_dir, str(label))
            if not os.path.isdir(folder):
                print(f"Skipping missing folder: {folder}")
                continue
                
            folder_count = 0
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(folder, file)
                    try:
                        img = Image.open(img_path)
                        binary = image_to_binary_string(img)
                        f.write(f"{binary},{label}\n")
                        folder_count += 1
                        count += 1
                    except Exception as e:
                        print(f"❌ Error processing image {img_path}: {e}")
            
            print(f"Processed {folder_count} images for digit {label}")
    
    print(f"✅ {output_file} generated successfully with {count} images!")

def build_all_datasets(data_dir="dataset"):
    """Build binary datasets for all available languages"""
    languages = []
    
    # Check which language directories exist
    for item in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, item)):
            if item in ["en", "bn"]: 
                languages.append(item)
    
    if not languages:
        print(f"❌ No language directories found in {data_dir}")
        return
        
    print(f"Found language directories: {', '.join(languages)}")
    
    for lang in languages:
        print(f"\nProcessing {lang} dataset...")
        build_binary_dataset(data_dir, lang)
    
    print("\nAll datasets processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert digit images to binary dataset")
    parser.add_argument("--dir", default="dataset", help="Base directory containing the dataset")
    parser.add_argument("--lang", default=None, help="Language code (en, bn, or 'all' for both)")
    parser.add_argument("--output", default=None, help="Output filename")
    
    args = parser.parse_args()
    
    if args.lang == "all" or args.lang is None:
        build_all_datasets(args.dir)
    else:
        build_binary_dataset(args.dir, args.lang, args.output)