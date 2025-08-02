import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder_path, image_size=(64, 64)):
    images = []
    labels = []
    class_names = os.listdir(folder_path)
    
    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                images.append(np.array(img).flatten())  # flatten to 1D
                labels.append(class_name)
            except:
                print(f"Failed to load image: {img_path}")
    
    return np.array(images), np.array(labels)

def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le