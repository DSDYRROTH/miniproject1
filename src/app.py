import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')

# Configure CORS
CORS(app, resources={r"/recommend": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}})

# Constants
PRICE_SCALER = 3000
CAPACITY_SCALER = 50

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', r'D:\mini_project1\miniproject1\models\space_recommender.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', r'D:\mini_project1\miniproject1\data\processed_spaces.csv')

# Load model and data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Get the min/max scaling from the data
price_max = df['price'].max()
capacity_max = df['capacity'].max()

@app.route('/')
def home():
    return render_template('map.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.get_json()

        required_fields = ['price', 'capacity']
        missing = [field for field in required_fields if field not in user_input]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        amenities = user_input.get('amenities', [])
        if not isinstance(amenities, list):
            return jsonify({"error": "Amenities should be a list"}), 400

        # Normalize the price and capacity
        normalized_price = float(user_input['price']) / price_max
        normalized_capacity = float(user_input['capacity']) / capacity_max

        input_data = [[
            normalized_price,
            normalized_capacity,
            1 if "AC" in amenities else 0,
            1 if "Fast Food" in amenities else 0
        ]]

        distances, indices = model.kneighbors(input_data)

        # Get the recommendations and include amenities and lat/lng
        recommendations = df.iloc[indices[0]].replace({np.nan: None}).to_dict('records')

        # Add the lat and lng to the recommendations
        for rec in recommendations:
            rec['amenities'] = rec.get('amenities', '').split(', ')  # Ensure amenities are formatted correctly
            rec['lat'] = float(rec['lat'])  # Ensure lat is float for map
            rec['lng'] = float(rec['lng'])  # Ensure lng is float for map

        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "count": len(recommendations)
        })

    except Exception as e:
        app.logger.error(f"Recommendation error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=os.getenv("FLASK_DEBUG", "False") == "True")
