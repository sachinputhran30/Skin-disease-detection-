import json
import os

BASE_DIR = os.path.dirname(__file__)

# Load class names from JSON file
JSON_PATH = os.path.join(BASE_DIR, "classes.json")

with open(JSON_PATH, "r") as f:
    classes_data = json.load(f)

DISEASE_CLASSES = classes_data["DISEASE_CLASSES"]
SKIN_CLASSES = classes_data["SKIN_CLASSES"]


# Preprocessing function
import tensorflow as tf
import numpy as np

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# Recommendations for disease
def get_recommendation(disease):
    rec = {
        "Acne": "Use salicylic acid or benzoyl peroxide. Keep the skin clean.",
        "Eczema": "Moisturize regularly and avoid harsh soaps.",
        "Melanoma": "Seek immediate dermatologist consultation.",
        "Psoriasis": "Use medicated creams and keep the skin moisturized.",
        "Vitiligo": "Consult a dermatologist for treatment options."
    }
    return rec.get(disease, "Consult a dermatologist.")


# Recommendations for skin type
def get_skin_recommendation(skin_type):
    rec = {
        "dry": "Use heavy moisturizers and avoid hot water.",
        "normal": "Maintain a balanced skincare routine.",
        "oily": "Use oil-free products and wash face twice daily."
    }
    return rec.get(skin_type, "Maintain a healthy skincare routine.")
