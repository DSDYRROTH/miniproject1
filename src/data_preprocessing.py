import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    # Read the CSV file directly (remove the redundant DataFrame creation)
    df = pd.read_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv")
    
    # Convert rating to float (only if needed)
    if df['rating'].dtype == object:
        df['rating'] = df['rating'].astype(float)

    # Standardize amenities names (only if they exist in the current format)
    amenities_mapping = {
        'Area': 'Indoor Seating Area',
        'street': 'Street Parking',
        'AC': 'Air Conditioning',
        'regional': 'Regional Cuisine',
        'fastfood': 'Fast Food'
    }
    df['amenities'] = df['amenities'].replace(amenities_mapping)

    # One-hot encode amenities (modified approach)
    # First get unique amenities if they're in a list format
    if df['amenities'].str.contains(',').any():
        amenities_dummies = df['amenities'].str.get_dummies(sep=',')
    else:
        amenities_dummies = pd.get_dummies(df['amenities'])
    
    # Add only missing amenity columns to avoid duplicates
    for col in amenities_dummies.columns:
        if col not in df.columns:
            df[col] = amenities_dummies[col]
        else:
            df[col] = df[col] | amenities_dummies[col]  # Combine if column exists

    # Normalize price and capacity
    scaler = MinMaxScaler()
    df[['price_norm', 'capacity_norm']] = scaler.fit_transform(df[['price', 'capacity']])
    
    # Save processed data
    df.to_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv", index=False)
    return df

if __name__ == "__main__":
    preprocess_data()