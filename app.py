import os
import json
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import webbrowser
from threading import Timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load JSON (class names)
with open(os.path.join(BASE_DIR, "classes.json"), "r") as f:
    class_data = json.load(f)

DISEASE_CLASSES = class_data["disease_classes"]
SKIN_TYPE_CLASSES = class_data["skin_type_classes"]

# Load Models
disease_model = load_model(os.path.join(BASE_DIR, "models/disease_model.h5"))
skin_type_model = load_model(os.path.join(BASE_DIR, "models/skintype_model.h5"))

# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# FIRST-LINE TREATMENT DATA
# ==============================
treatment_info = {
    "Acne": (
        "Cleanse the face twice daily using a mild cleanser. "
        "Use topical agents such as benzoyl peroxide or salicylic acid. "
        "Avoid oily cosmetics and do not squeeze pimples."
    ),

    "Eczema": (
        "Apply fragrance-free moisturizers multiple times a day. "
        "Avoid harsh soaps and allergens. "
        "Mild topical corticosteroids may be used if prescribed."
    ),

    "Psoriasis": (
        "Keep the skin well moisturized. "
        "Apply topical corticosteroids or vitamin D analog creams. "
        "Avoid known triggers such as stress and skin injury."
    ),

    "Melanoma": (
        "Immediate consultation with a dermatologist is essential. "
        "Do not attempt self-treatment. "
        "Early medical intervention is critical."
    ),

    "Vitiligo": (
        "Consult a dermatologist for proper evaluation. "
        "Topical corticosteroids or calcineurin inhibitors may be recommended. "
        "Sun protection is important."
    ),

    "Healthy": (
        "Your skin appears healthy. "
        "Maintain good skincare practices, regular cleansing, and sunscreen use."
    )
}

skin_care_recommendations = {
    "Oily": (
        "Use a gentle foaming cleanser twice daily. "
        "Choose oil-free and non-comedogenic products. "
        "Use a lightweight gel-based moisturizer."
    ),

    "Dry": (
        "Use a mild, hydrating cleanser. "
        "Apply thick moisturizers regularly. "
        "Avoid hot water and harsh soaps."
    ),

    "Normal": (
        "Maintain regular cleansing and moisturizing. "
        "Use sunscreen daily. "
        "Follow a balanced skincare routine."
    )
}
skin_care_products = {
    "Oily": [
        {
            "name": "Minimalist Salicylic Acid Cleanser",
            "image": "https://m.media-amazon.com/images/I/51n6kTG3zQL._SL1200_.jpg",
            "link": "https://amzn.in/d/1Iglb4z/"
        },
        {
            "name": "Neutrogena Hydro Boost Gel",
            "image": "https://m.media-amazon.com/images/I/51YTdG8RPSL._SL1001_.jpg",
            "link": "https://amzn.in/d/6Rl8qny"
        }
    ],

    "Dry": [
        {
            "name": "Cetaphil Gentle Cleanser",
            "image": "https://m.media-amazon.com/images/I/41kd85yMQnL._SY300_SX300_QL70_FMwebp_.jpg",
            "link": "https://amzn.in/d/f3o7r6f"
        },
        {
            "name": "Nivea Nourishing Cream",
            "image": "https://m.media-amazon.com/images/I/4125gF+XmHL._SL1000_.jpg",
            "link": "https://amzn.in/d/boRlgiL"
        }
    ],

    "Normal": [
        {
            "name": "Simple Refreshing Face Wash",
            "image": "https://m.media-amazon.com/images/I/41p2stkT-4L._SY300_SX300_QL70_FMwebp_.jpg",
            "link": "https://amzn.in/d/21uSt0I"
        },
        {
            "name": "Lakme Peach Milk Moisturizer",
            "image": "https://m.media-amazon.com/images/I/51TWJsDasqL._SL1000_.jpg",
            "link": "https://amzn.in/d/c7ecL5P"
        }
    ]
    
}


dos_donts = {
    "Acne": {
        "dos": [
            "Wash face twice daily with a mild cleanser",
            "Use oil-free and non-comedogenic products",
            "Apply prescribed topical medications",
            "Drink plenty of water",
            "Maintain a balanced diet"
        ],
        "donts": [
            "Do not squeeze or pop pimples",
            "Avoid touching your face frequently",
            "Do not use oily cosmetics",
            "Avoid excessive sun exposure",
            "Do not skip skincare routine"
        ]
    },

    "Eczema": {
        "dos": [
            "Apply moisturizer multiple times a day",
            "Use fragrance-free skincare products",
            "Wear soft cotton clothing",
            "Keep skin hydrated",
            "Follow dermatologist advice"
        ],
        "donts": [
            "Avoid scratching affected areas",
            "Do not use harsh soaps",
            "Avoid allergens",
            "Avoid very hot showers",
            "Do not ignore flare-ups"
        ]
    },

    "Psoriasis": {
        "dos": [
            "Keep skin well moisturized",
            "Use prescribed creams regularly",
            "Reduce stress levels",
            "Maintain healthy lifestyle",
            "Get proper sleep"
        ],
        "donts": [
            "Avoid skin injuries",
            "Do not smoke or consume alcohol",
            "Avoid harsh chemical products",
            "Do not skip treatment",
            "Avoid dry environments"
        ]
    },

    "Vitiligo": {
        "dos": [
            "Use sunscreen daily",
            "Follow medical treatment",
            "Maintain skin hygiene",
            "Eat nutritious food",
            "Stay mentally positive"
        ],
        "donts": [
            "Avoid skin trauma",
            "Do not use unprescribed creams",
            "Avoid excessive sun exposure",
            "Avoid chemical cosmetics",
            "Do not stress over appearance"
        ]
    },

    "Melanoma": {
        "dos": [
            "Consult dermatologist immediately",
            "Monitor skin changes regularly",
            "Use high SPF sunscreen",
            "Wear protective clothing",
            "Follow treatment strictly"
        ],
        "donts": [
            "Do not self-medicate",
            "Avoid tanning beds",
            "Do not ignore suspicious moles",
            "Avoid prolonged sun exposure",
            "Do not delay diagnosis"
        ]
    },

    "Healthy": {
        "dos": [
            "Maintain daily cleansing routine",
            "Use moisturizer regularly",
            "Apply sunscreen daily",
            "Drink enough water",
            "Get proper sleep"
        ],
        "donts": [
            "Avoid excessive sun exposure",
            "Do not skip skincare routine",
            "Avoid harsh products",
            "Do not over-exfoliate",
            "Avoid using expired products"
        ]
    }
}
skin_routine = {
    "Oily": {
        "morning": [
            "Wash face with a gentle foaming cleanser",
            "Apply oil-free moisturizer",
            "Use non-comedogenic sunscreen"
        ],
        "night": [
            "Cleanse face to remove oil and dirt",
            "Apply salicylic acid or niacinamide serum",
            "Use a lightweight gel-based moisturizer"
        ]
    },

    "Dry": {
        "morning": [
            "Use a hydrating cream cleanser",
            "Apply thick moisturizer",
            "Use sunscreen with SPF 30+"
        ],
        "night": [
            "Cleanse with mild cleanser",
            "Apply hydrating serum (hyaluronic acid)",
            "Use a nourishing night cream"
        ]
    },

    "Normal": {
        "morning": [
            "Cleanse face with mild cleanser",
            "Apply light moisturizer",
            "Use sunscreen daily"
        ],
        "night": [
            "Cleanse to remove dirt and makeup",
            "Apply nourishing serum",
            "Use a gentle night moisturizer"
        ]
    }
}
skin_routine = { "Oily": { "morning": [ "Wash face with a gentle foaming cleanser", "Apply oil-free moisturizer", "Use non-comedogenic sunscreen" ], "night": [ "Cleanse face to remove oil and dirt", "Apply salicylic acid or niacinamide serum", "Use a lightweight gel-based moisturizer" ] }, "Dry": { "morning": [ "Use a hydrating cream cleanser", "Apply thick moisturizer", "Use sunscreen with SPF 30+" ], "night": [ "Cleanse with mild cleanser", "Apply hydrating serum (hyaluronic acid)", "Use a nourishing night cream" ] }, "Normal": { "morning": [ "Cleanse face with mild cleanser", "Apply light moisturizer", "Use sunscreen daily" ], "night": [ "Cleanse to remove dirt and makeup", "Apply nourishing serum", "Use a gentle night moisturizer" ] } }


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/disease")
def disease_page():
    return render_template("disease.html")


@app.route("/skin_type")
def skin_type_page():
    return render_template("skin_type.html")


@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    img = preprocess_image(img_path)
    preds = disease_model.predict(img)[0]

    index = int(np.argmax(preds))
    result = DISEASE_CLASSES[index]
    confidence = float(preds[index]) * 100

    # Retrieve treatment
    treatment = treatment_info.get(
        result, "No treatment information available."
    )

    # Do's and Don'ts
    do_dont = dos_donts.get(
        result,
        {
            "dos": "Follow doctor advice.",
            "donts": "Avoid self medication."
        }
    )

    return render_template(
        "disease_result.html",
        result=result,
        confidence=round(confidence, 2),
        treatment=treatment,
        dos=do_dont["dos"],
        donts=do_dont["donts"],
        img_path="static/uploads/" + file.filename
    )

  
@app.route("/predict_skin_type", methods=["POST"])
def predict_skin_type():
    file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    img_arr = preprocess_image(img_path)
    preds = skin_type_model.predict(img_arr)[0]

    idx = int(np.argmax(preds))
    result = SKIN_TYPE_CLASSES[idx].title()
    # confidence = float(preds[idx]) * 100
    recommendation = skin_care_recommendations.get(
    result, "Follow a basic skincare routine."
)

    products = skin_care_products.get(result, [])
    routine = skin_routine.get(result, {"morning": [], "night": []})
    return render_template(
    "skin_type_result.html",
    result=result,
    # confidence=round(confidence, 2),
    recommendation=recommendation,
    products=products,
    routine=routine,
    img_path="static/uploads/" + file.filename    
)

# Auto-open Browser
def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)
