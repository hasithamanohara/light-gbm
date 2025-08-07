from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from asgiref.wsgi import WsgiToAsgi

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('./recomondation/restaurant_recommender.pkl')
    scaler = joblib.load('./recomondation/feature_scaler.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure 'restaurant_recommender.pkl' and 'feature_scaler.pkl' are in the same directory.")
    exit(1)

# Define the features used in the model (from the notebook)
features = [
    'user_sentiment_pref', 'user_price_pref', 'user_proximity_pref',
    'sentiment_score', 'price_score', 'proximity_score',
    'sentiment_diff', 'price_diff', 'proximity_diff'
]

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'user' not in data or 'restaurants' not in data:
            return jsonify({'error': 'Invalid input. Expect "user" and "restaurants" in JSON body.'}), 400
        
        user = data['user']
        restaurants = data['restaurants']
        
        # Validate user preferences
        for key in ['user_sentiment_pref', 'user_price_pref', 'user_proximity_pref']:
            if not isinstance(user.get(key), (int, float)) or not 0.1 <= user[key] <= 0.99:
                return jsonify({'error': f'{key} must be a float between 0.1 and 0.99'}), 400
        
        # Validate restaurants
        if not restaurants:
            return jsonify({'error': 'Restaurants list cannot be empty'}), 400
        for r in restaurants:
            for key in ['sentiment_score', 'price_score', 'proximity_score']:
                if not isinstance(r.get(key), (int, float)) or not 0.1 <= r[key] <= 0.99:
                    return jsonify({'error': f'{key} in restaurant {r.get("restaurant_id")} must be a float between 0.1 and 0.99'}), 400
        
        # Create DataFrame for user (repeat user prefs for each restaurant)
        user_df = pd.DataFrame([user] * len(restaurants))
        
        # Create DataFrame for restaurants
        restaurants_df = pd.DataFrame(restaurants)
        
        # Combine into pairs
        pairs = pd.concat([user_df, restaurants_df], axis=1)
        
        # Feature engineering
        pairs['sentiment_diff'] = abs(pairs['user_sentiment_pref'] - pairs['sentiment_score'])
        pairs['price_diff'] = abs(pairs['user_price_pref'] - pairs['price_score'])
        pairs['proximity_diff'] = abs(pairs['user_proximity_pref'] - pairs['proximity_score'])
        
        # Prepare features for prediction
        X_new = pairs[features]
        X_new_scaled = scaler.transform(X_new)
        
        # Predict relevance
        predictions = model.predict(X_new_scaled)
        pairs['predicted_relevance'] = np.round(predictions).clip(1, 5)
        
        # Sort by predicted_relevance descending and get top 10
        top_recs = pairs.sort_values('predicted_relevance', ascending=False).head(10)
        
        # Return as JSON
        return jsonify(top_recs.to_dict(orient='records'))
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Wrap Flask app for ASGI compatibility
asgi_app = WsgiToAsgi(app)