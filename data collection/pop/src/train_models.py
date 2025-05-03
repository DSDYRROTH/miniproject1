import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

def train_retail_recommendation_model():
    # Load the processed retail data
    df = pd.read_csv(r"d:/mini project/data preprocessing/data collection/pop/data/processed_shop.csv")
    
    # Select relevant features from the data
    feature_columns = [
        'price_norm', 
        'capacity_norm',
        'Air Conditioning',
        'Fast Fashion',
        'Indoor Shopping Area',
        'Regional Brands',
        'Street Access'
    ]
    
    # Ensure we only use columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    features = df[available_features]
    
    # Train KNN model
    n_neighbors = min(5, len(df))  # Ensure we don't request more neighbors than samples
    model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    model.fit(features)
    
    # Save model
    model_path = r"d:/mini project/data preprocessing/data collection/pop/models/retail_space_recommender.pkl"
    joblib.dump(model, model_path)
    print(f"Retail recommendation model trained and saved to {model_path}")
    
    # Save the feature columns for reference during prediction
    joblib.dump(available_features, r"d:/mini project/data preprocessing/data collection/pop/models/retail_feature_columns.pkl")
    print("Feature columns metadata saved")

if __name__ == "__main__":
    train_retail_recommendation_model()
