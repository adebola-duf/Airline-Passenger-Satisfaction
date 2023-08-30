import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import pickle
from scipy.stats import boxcox


st.title("Airline Passenger Satisfaction")

image = Image.open('airplane.png')

st.image(image)

gender = st.selectbox("Choose sex", ['Male', 'Female'])
gender = 1 if gender == 'Male' else 0

flying_class = st.selectbox("What class did you fly in?", [
                            'Business', 'Economy', 'Economy Plus'])
if flying_class == 'Business':
    class_eco = 0
    class_eco_plus = 0
elif flying_class == 'Economy':
    class_eco = 1
    class_eco_plus = 0
else:
    class_eco = 0
    class_eco_plus = 1


customer_type = st.selectbox("Choose Customer Type", [
                             'Loyal Customer', 'Disloyal Customer'])
customer_type = 1 if customer_type == 'Loyal Customer' else 0

age = st.slider("Choose age", 1, 100)

type_travel = st.selectbox("What was the purpose of your travel?", [
    'Business Travel', 'Personal Travel'])
type_travel = 1 if type_travel == 'Business Travel' else 0

flight_distance = st.slider("What's the flight distance?", 10, 10000)

inflight_wifi_service = st.slider(
    "How do you rate the wifi service in the flight? Ps: 0 means you don't want to rate it.", 0, 5)

departure_arrival_time_convenient = st.slider(
    "How do you rate the departure or arrival time convenience?", 0, 5)

ease_of_online_booking = st.slider(
    "How do was the online booking process?", 0, 5)

gate_location = st.slider(
    "What's your satisfaction level of our gate location?", 0, 5)

food_and_drink = st.slider(
    "How did you like the food and drinks served on board?", 0, 5)

online_boarding = st.slider(
    "What's your satisfaction level for online boarding?", 0, 5)

seat_comfort = st.slider("How comfortable was your seat?", 0, 5)

inflight_entertainment = st.slider(
    "How did you like the entertainment in the plane?", 0, 5)

onboard_service = st.slider("How was the onboard service?", 0, 5)

leg_room_service = st.slider("How comfortable was the leg room for you?", 0, 5)

baggage_handling = st.slider("How well were your baggages handled?", 0, 5)

check_in_service = st.slider("How did you like our checkin services?", 0, 5)

inflight_service = st.slider(
    "How much did you like the inflight service?", 0, 5)

cleanliness = st.slider("How clean was the airplane?", 0, 5)

departure_delay = st.slider(
    "For how long was your departure delayed in Minutes? ", 0, 2000)

arrival_delay = st.slider(
    "For how long was your arrival delayed in Minutes? ", 0, 2000)

user_data = np.array([gender, customer_type, age, type_travel, flight_distance, inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding,
                     seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, check_in_service, inflight_service, cleanliness, departure_delay, arrival_delay, class_eco, class_eco_plus])

user_data = pd.Series(data=user_data, index=['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
                      'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'Class_Eco', 'Class_Eco Plus'])
user_data = pd.DataFrame(user_data).T

with open('Model and Transformers.pickle', 'rb') as m:
    loaded_data = pickle.load(m)

loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']
loaded_lambdas = loaded_data['lambdas']

for column in user_data:
    if column in loaded_lambdas:
        user_data[column] = boxcox(
            user_data[column] + 1, loaded_lambdas[column])
user_data = loaded_scaler.transform(user_data)


def predict(user_data):
    satisfaction = loaded_model.predict(user_data)
    if satisfaction == 1:
        st.success('Customer is satisfied with our airline:thumbsup:')
    else:
        st.error('Customer is not satisfied with our airline :thumbsdown:')


if st.button('Predict'):
    predict(user_data)
