from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Update your model path and load the model
house_price_model_path = 'model03.pkl'  # Replace with your model path
with open(house_price_model_path, 'rb') as file:
    house_price_model = pickle.load(file)

app = Flask(__name__)

# Define the route for prediction
@app.route('/house_price_predict', methods=['POST'])
def house_price_predict():
    # Extract data from form for house price prediction
    avg_area_income = request.form.get('Avg. Area Income', type=float)
    avg_area_house_age = request.form.get('Avg. Area House Age', type=float)
    avg_area_num_rooms = request.form.get('Avg. Area Number of Rooms', type=float)
    avg_area_num_bedrooms = request.form.get('Avg. Area Number of Bedrooms', type=float)
    area_population = request.form.get('Area Population', type=float)

    # Validate inputs
    if (avg_area_income is None or avg_area_house_age is None or
        avg_area_num_rooms is None or avg_area_num_bedrooms is None or
        area_population is None):
        return render_template('house_price.html', prediction_text='Invalid input. Please provide all fields.')

    # Create numpy array for prediction
    final_features = np.array([[avg_area_income, avg_area_house_age,
                                avg_area_num_rooms, avg_area_num_bedrooms,
                                area_population]])

    # Make house price prediction
    predicted_price = house_price_model.predict(final_features)[0]

    return render_template('house_price.html', prediction_text='Predicted House Price: ${:.2f}'.format(predicted_price))

# Define another route for the main page
@app.route('/')
def main_page():
    return render_template('house_price.html')

if __name__ == "__main__":
    app.run(debug=True)
