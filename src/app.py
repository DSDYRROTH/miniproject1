from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = joblib.load(r"D:\mini_project1\miniproject1\models\space_recommender.pkl")
df = pd.read_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv")

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json  # e.g., {"price": 2000, "capacity": 15, "amenities": ["AC"]}
    # Preprocess input
    input_data = [[
        user_input['price'] / 3000,  # Manual normalization (adjust max price)
        user_input['capacity'] / 50,  # Manual normalization (adjust max capacity)
        1 if "AC" in user_input['amenities'] else 0,
        1 if "Parking" in user_input['amenities'] else 0,
        1 if "Indoor" in user_input['amenities'] else 0
    ]]
    # Get recommendations
    distances, indices = model.kneighbors(input_data)
    recommendations = df.iloc[indices[0]].to_dict('records')
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True,port=5001)