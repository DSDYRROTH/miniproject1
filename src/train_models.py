import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

def train_recommendation_model():
    df = pd.read_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv")
    
    # Ensure we're using the same features as in app.py
    features = df[['price_norm', 'capacity_norm', 'Air Conditioning', 'Fast Food']]
    
    # Train KNN
    model = NearestNeighbors(n_neighbors=min(3, len(df)), metric='cosine')  # Handle small datasets
    model.fit(features)
    
    # Save model
    joblib.dump(model, r"D:\mini_project1\miniproject1\models\space_recommender.pkl")
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_recommendation_model()