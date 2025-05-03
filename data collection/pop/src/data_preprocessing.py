import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    
    df = pd.read_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv")

    if df['rating'].dtype == object:
        df['rating'] = df['rating'].astype(float)

    
    amenities_mapping = {
        'Area': 'Indoor Seating Area',
        'street': 'Street Parking',
        'AC': 'Air Conditioning',
        'regional': 'Regional Cuisine',
        'fastfood': 'Fast Food'
    }
    df['amenities'] = df['amenities'].replace(amenities_mapping)

    
    if df['amenities'].str.contains(',').any():
        amenities_dummies = df['amenities'].str.get_dummies(sep=',')
    else:
        amenities_dummies = pd.get_dummies(df['amenities'])
    
    
    for col in amenities_dummies.columns:
        if col not in df.columns:
            df[col] = amenities_dummies[col]
        else:
            df[col] = df[col] | amenities_dummies[col] 

    
    scaler = MinMaxScaler()
    df[['price_norm', 'capacity_norm']] = scaler.fit_transform(df[['price', 'capacity']])
    
    # Save processed data
    df.to_csv(r"D:\mini_project1\miniproject1\data\processed_spaces.csv", index=False)
    return df

if __name__ == "__main__":
    preprocess_data()