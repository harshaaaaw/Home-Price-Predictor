import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load dataset and model
data = pd.read_csv('train.csv')
model = pickle.load(open('model.pkl', 'rb'))

# Encode the location using LabelEncoder
le = LabelEncoder()
data['Location_Encoded'] = le.fit_transform(data['Location'])

@app.route('/')
def index():
    locations = sorted(data['Location'].unique())
    
    # Prepare the data for the bar chart
    historical_data = pd.read_csv('Historical_Price_Trends.csv')
    future_data = pd.read_csv('Future_Price_Predictions.csv')
    
    combined_data = pd.concat([historical_data, future_data])
    
    years = combined_data['Year'].tolist()
    prices = combined_data['Price'].tolist()
    
    return render_template('home.html', locations=locations, years=years, prices=prices)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        area = float(request.form.get('area'))
        bhk = int(request.form.get('bhk'))
        
        # Collect other inputs, use 0 if not checked
        new_resale = 1 if request.form.get('toggle') == 'on' else 0
        gym = 1 if request.form.get('gym') == 'on' else 0
        car_parking = 1 if request.form.get('car_parking') == 'on' else 0
        indoor_games = 1 if request.form.get('indoor_games') == 'on' else 0
        jogging_track = 1 if request.form.get('jogging_track') == 'on' else 0
        
        # Find the encoded value for the location
        loc_index = le.transform([location])[0]
        
        # Prepare the input data for the model (8 features)
        input_data = np.array([[area, loc_index, bhk, new_resale, gym, car_parking, indoor_games, jogging_track]])
        
        # Predict the price (assuming the model output is in millions)
        pred = model.predict(input_data)[0] * 1e6  # Convert prediction to full rupee value
        
        # Log the predicted value for debugging
        print(f"Predicted Price (full number): {pred}")
        
        # Sample data for historical and future prices
        years = ['2021', '2022', '2023', '2024', '2025', '2026']
        historicalPrices = [120000, 150000, 180000, 200000, 220000, 250000]
        futurePrices = [140000, 170000, 190000, 210000, 240000, 270000]

        return {
            'prediction': str(int(pred)),  # Return the prediction as a full number
            'years': years,
            'historicalPrices': historicalPrices,
            'futurePrices': futurePrices
        }
    
    except Exception as e:
        print(f"Error during prediction: {e}")  # Print the error for debugging
        return f"Error: {e}"

    
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)
