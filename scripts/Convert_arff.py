import os
import numpy as np
from PIL import Image

def collect_image_data(dataset_path):
    data = []
    labels = []
    
    for language in ['en', 'bn']:
        for digit in range(10):
            folder = os.path.join(dataset_path, language, str(digit))
            if not os.path.exists(folder):
                continue
            
            for filename in os.listdir(folder):
                if filename.endswith('.png'):
                    filepath = os.path.join(folder, filename)
                    try:
                        img = Image.open(filepath).convert('L')  
                        img = img.resize((28,28))  
                        pixels = np.array(img)
                        
                        binary_pixels = (pixels <= 127).astype(int).flatten()
                        
                        if len(binary_pixels) == 0:
                            print(f"Warning: Empty image {filepath}")
                            continue
                            
                        data.append(binary_pixels)
                        labels.append(f"{language}_{digit}")
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        continue
    
    return data, labels

def write_arff(data, labels, output_file):
    if not data:
        raise ValueError("No data to write to ARFF file")
    
    num_pixels = len(data[0])
    
    for i, vector in enumerate(data):
        if len(vector) != num_pixels:
            raise ValueError(f"Inconsistent vector length at index {i}: expected {num_pixels}, got {len(vector)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("@RELATION digit_images\n\n")
        
        for i in range(num_pixels):
            f.write(f"@ATTRIBUTE pixel{i} {{0,1}}\n")
        
        class_list = sorted(list(set(labels)))
        f.write(f"@ATTRIBUTE class {{{','.join(class_list)}}}\n\n")
        
        f.write("@DATA\n")
        
        for vector, label in zip(data, labels):
            vector_str = ",".join(str(int(p)) for p in vector)
            f.write(f"{vector_str},{label}\n")

def validate_arff_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data_section_started = False
        line_count = 0
        invalid_values = 0
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if not data_section_started and not line:
                continue
                
            if line == "@DATA":
                data_section_started = True
                continue
            
            if data_section_started and line:
                line_count += 1
                parts = line.split(',')
                if len(parts) > 1:
                    pixel_values = parts[:-1]
                    for pixel in pixel_values:
                        if pixel not in ['0', '1']:
                            invalid_values += 1
                            break
        
        print(f"Validation complete: {line_count} data instances found")
        if invalid_values > 0:
            print(f"Warning: {invalid_values} instances with non-binary values")
        else:
            print("All pixel values are binary (0 or 1)")
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def print_dataset_statistics(data, labels):
    print("\nDataset Statistics:")
    print(f"Total instances: {len(data)}")
    
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Class distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]} instances")
    
    if data:
        total_pixels = len(data[0])
        all_pixels = np.array(data)
        ones_ratio = np.mean(all_pixels)
        print(f"Average pixel density (1s): {ones_ratio:.3f}")
        print(f"Total pixels per image: {total_pixels}")

if __name__ == "__main__":
    dataset_path = "dataset"  
    output_file = "digit_images.arff"
    
    try:
        print("Collecting image data...")
        data, labels = collect_image_data(dataset_path)
        
        if not data:
            print("No data collected. Please check your dataset path and structure.")
            print("Expected structure: dataset/[en|bn]/[0-9]/image.png")
            exit(1)
        
        print_dataset_statistics(data, labels)
        
        print(f"\nWriting ARFF file to {output_file}...")
        write_arff(data, labels, output_file)
        
        print("Validating ARFF file...")
        if validate_arff_file(output_file):
            print("ARFF file created successfully!")
            print(f"File saved as: {output_file}")
        else:
            print("ARFF file may have issues.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()