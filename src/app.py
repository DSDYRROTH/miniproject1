from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and data
model = joblib.load(r"D:\mini_project1\miniproject1\models\space_recommender.pkl")
df = pd.read_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.get_json()
        
        # Validate input
        required_fields = ['price', 'capacity']
        if not all(field in user_input for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Prepare features (MUST match training features exactly)
        input_data = [[
            float(user_input['price']) / 3000,  # Normalized price
            float(user_input['capacity']) / 50,  # Normalized capacity
            1 if "AC" in user_input.get('amenities', []) else 0,
            1 if "Fast Food" in user_input.get('amenities', []) else 0
        ]]
        
        # Get recommendations
        distances, indices = model.kneighbors(input_data)
        recommendations = df.iloc[indices[0]].to_dict('records')
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)