import joblib
import numpy as np
from PIL import Image

def predict_image(image_path, model_path='../models/fruit_classifier.pkl', image_size=(64, 64)):
    model, le = joblib.load(model_path)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img).flatten().reshape(1, -1)

    prediction = model.predict(img_array)
    predicted_label = le.inverse_transform(prediction)[0]
    return predicted_label
