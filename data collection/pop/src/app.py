import json
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import logging
from dotenv import load_dotenv
import os
from sklearn.neighbors import NearestNeighbors
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Load environment variables
load_dotenv()

# Create a Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define amenities lists
amenities_list_retail = ['Air Conditioning', 'Fast Fashion', 'Indoor Shopping Area', 'Regional Brands', 'Street Access']
amenities_list_food = ['Air Conditioning', 'Fast Food', 'Indoor Seating Area', 'Regional Cuisine', 'Street Parking']
# Subset of amenities for food model (assuming model_food expects 2 amenities)
amenities_list_food_model = ['Air Conditioning', 'Fast Food']  # Adjust based on actual training features

# Load retail and food data
try:
    retail_data = pd.read_csv(r"D:\mini project\data preprocessing\data collection\pop\src\processed_shop.csv")
    food_data = pd.read_csv(r"D:\mini project\data preprocessing\data collection\pop\src\processed_spaces.csv")
    logger.info("Retail and food data loaded successfully")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

# Simulate historical price data if not present
def add_historical_prices(df):
    if 'date' not in df.columns:
        # Generate synthetic historical data (30 days for each space)
        dates = pd.date_range(end='2025-04-28', periods=30, freq='D')
        historical_data = []
        for idx, row in df.iterrows():
            base_price = row['price']
            # Simulate price fluctuations (±10% noise)
            prices = base_price + np.random.normal(0, base_price * 0.1, 30)
            for date, price in zip(dates, prices):
                historical_data.append({
                    'title': row['title'],
                    'date': date,
                    'price': max(0, price)  # Ensure non-negative prices
                })
        historical_df = pd.DataFrame(historical_data)
        return historical_df
    return df

retail_historical = add_historical_prices(retail_data)
food_historical = add_historical_prices(food_data)

# Normalize price and capacity if not present
def normalize_columns(df, price_col='price', capacity_col='capacity'):
    if 'price_norm' not in df.columns:
        df['price_norm'] = df[price_col] / df[price_col].max()
    if 'capacity_norm' not in df.columns:
        df['capacity_norm'] = df[capacity_col] / df[capacity_col].max()
    return df

retail_data = normalize_columns(retail_data)
food_data = normalize_columns(food_data)

# Ensure binary amenities columns exist
for amenity in amenities_list_retail:
    if amenity not in retail_data.columns:
        retail_data[amenity] = retail_data['amenities'].apply(lambda x: 1 if amenity in str(x).split(', ') else 0)
for amenity in amenities_list_food:
    if amenity not in food_data.columns:
        food_data[amenity] = food_data['amenities'].apply(lambda x: 1 if amenity in str(x).split(', ') else 0)

# Load pre-trained models
try:
    model_retail = joblib.load(r'D:\mini project\data preprocessing\data collection\pop\models\retail_space_recommender.pkl')
    model_food = joblib.load(r'D:\mini project\data preprocessing\data collection\pop\models\space_recommender.pkl')
    logger.info(f"model_retail type: {type(model_retail)}")
    logger.info(f"model_food type: {type(model_food)}")
    if not isinstance(model_retail, NearestNeighbors) or not isinstance(model_food, NearestNeighbors):
        raise ValueError("Loaded models are not NearestNeighbors instances")
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Create database tables
with app.app_context():
    db.create_all()
sales_data = {
    'sales': [100, 150, 130, 170, 160, 180, 190, 200, 210, 220],
}
date_range = pd.date_range(start='2025-01-01', periods=10, freq='D')
sales_df = pd.DataFrame(sales_data, index=date_range)
@app.route('/')
def index():
    username = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        username = user.username if user else None
    return render_template('index.html', username=username)

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/pricing.html')
def pricing_page():
    return render_template('pricing.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        business_type = data['business_type']
        price = int(data['price'])
        capacity = int(data['capacity'])
        amenities = data['amenities']

        if business_type == 'retail':
            amenities_list = amenities_list_retail
            model = model_retail
            shop_data = retail_data
            price_max = shop_data['price'].max()
            capacity_max = shop_data['capacity'].max()
        elif business_type == 'food':
            amenities_list = amenities_list_food_model  # Use subset for model input
            model = model_food
            shop_data = food_data
            price_max = shop_data['price'].max()
            capacity_max = shop_data['capacity'].max()
        else:
            return jsonify({'status': 'error', 'error': 'Invalid business type'}), 400

        price_norm = price / price_max
        capacity_norm = capacity / capacity_max
        input_features = np.array([price_norm, capacity_norm] + 
                                 [1 if amenity in amenities else 0 for amenity in amenities_list])

        logger.info(f"Input features shape: {input_features.shape}, features: {input_features}")

        distances, indices = model.kneighbors([input_features])
        recommendations = []
        for idx in indices[0]:
            space = shop_data.iloc[idx]
            lng = float(space['lng']) if business_type == 'food' else float(space['long'])

            recommendations.append({
                'title': space['title'],
                'price': float(space['price']),
                'capacity': int(space['capacity']),
                'amenities': space['amenities'].split(', ') if isinstance(space['amenities'], str) else [],
                'lat': float(space['lat']),
                'lng': lng,
                'rating': float(space['rating']) if pd.notna(space['rating']) else 'Not rated'
            })

        return jsonify({'status': 'success', 'recommendations': recommendations})
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/forecast_price', methods=['POST'])
def forecast_price():
    try:
        data = request.get_json()
        business_type = data['business_type']
        space_title = data['space_title']

        if business_type == 'retail':
            historical_data = retail_historical
        elif business_type == 'food':
            historical_data = food_historical
        else:
            return jsonify({'status': 'error', 'error': 'Invalid business type'}), 400

        # Filter historical data for the selected space
        space_data = historical_data[historical_data['title'] == space_title]
        if space_data.empty:
            return jsonify({'status': 'error', 'error': 'No historical data for this space'}), 404

        # Ensure data is sorted by date
        space_data = space_data.sort_values('date')
        price_series = space_data.set_index('date')['price']

        # Prepare historical data for response
        historical_prices = [
            {'date': date.strftime('%Y-%m-%d'), 'price': float(price)}
            for date, price in price_series.items()
        ]

        # Check if enough data points
        if len(price_series) < 3:
            return jsonify({'status': 'error', 'error': f'Not enough data points. Need at least 3, found {len(price_series)}'}), 400

        # Fit ARIMA model with simpler parameters to reduce volatility
        model = ARIMA(price_series, order=(1, 1, 0))
        model_fit = model.fit()

        # Forecast next 30 days
        forecast_steps = 30
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_dates = pd.date_range(start=space_data['date'].iloc[-1] + timedelta(days=1), periods=forecast_steps, freq='D')

        # Smooth forecast to limit daily price changes (±5% of last historical price)
        last_historical_price = price_series.iloc[-1]
        max_change = last_historical_price * 0.05  # 5% max daily change
        smoothed_forecast = []
        prev_price = last_historical_price
        for price in forecast:
            # Cap the price change
            if price > prev_price + max_change:
                price = prev_price + max_change
            elif price < prev_price - max_change:
                price = prev_price - max_change
            smoothed_forecast.append(price)
            prev_price = price

        # Prepare forecast data
        forecast_results = [
            {'date': date.strftime('%Y-%m-%d'), 'price': float(price)}
            for date, price in zip(forecast_dates, smoothed_forecast)
        ]

        return jsonify({
            'status': 'success',
            'historical': historical_prices,
            'forecast': forecast_results
        })
    except Exception as e:
        logger.error(f"Error in price forecasting: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            return render_template('signup.html', error="All fields are required")

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        if User.query.filter_by(username=username).first():
            return render_template('signup.html', error="Username already exists")

        if User.query.filter_by(email=email).first():
            return render_template('signup.html', error="Email already exists")

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session['user_id'] = user.id
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))


@app.route('/forecast_sales', methods=['POST'])
def forecast_sales():
    try:
        data = request.get_json()
        start_date = data['start_date']
        end_date = data['end_date']
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            if start_date >= end_date:
                return jsonify({'status': 'error', 'error': 'Start date must be earlier than end date'}), 400
        except ValueError:
            return jsonify({'status': 'error', 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        df_filtered = sales_df[start_date:end_date]
        if len(df_filtered) < 3:
            return jsonify({'status': 'error', 'error': f'Not enough data points. Need at least 3, found {len(df_filtered)}'}), 400

        model = ARIMA(df_filtered['sales'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast_steps = 5
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_dates = pd.date_range(start=end_date + timedelta(days=1), periods=forecast_steps, freq='D')

        historical_sales = [
            {'date': date.strftime('%Y-%m-%d'), 'sales': float(sales)}
            for date, sales in df_filtered['sales'].items()
        ]
        forecast_sales = [
            {'date': date.strftime('%Y-%m-%d'), 'sales': float(sales)}
            for date, sales in zip(forecast_dates, forecast)
        ]

        return jsonify({
            'status': 'success',
            'historical': historical_sales,
            'forecast': forecast_sales
        })
    except Exception as e:
        logger.error(f"Error in sales forecasting: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)