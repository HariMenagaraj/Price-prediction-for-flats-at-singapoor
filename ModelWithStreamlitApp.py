import pickle
import pandas as pd
import streamlit as st
from PIL import Image

def load_model(model_file):
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def convert_storey_range(storey_range):
    return (int(storey_range.split(' TO ')[0]) + int(storey_range.split(' TO ')[1])) / 2

flat_type_mapping = {'1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4, '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI GENERATION': 7}

month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

storey = ['01 TO 03', '01 TO 05', '04 TO 06', '06 TO 10', '07 TO 09',
          '10 TO 12', '11 TO 15', '13 TO 15', '16 TO 18', '16 TO 20',
          '19 TO 21', '21 TO 25', '22 TO 24', '25 TO 27', '26 TO 30',
          '28 TO 30', '31 TO 33', '31 TO 35', '34 TO 36', '36 TO 40',
          '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51']

all_towns = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
             'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG',
             'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
             'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
             'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES',
             'TOA PAYOH', 'WOODLANDS', 'YISHUN']

all_flat_models = ['2-Room', '3Gen', 'Adjoined Flat', 'Apartment', 'Dbss',
                   'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette',
                   'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment', 'Premium Apartment Loft',
                   'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']

# Set the page configuration
st.set_page_config(layout="wide")

# Custom CSS for background color and other styles
st.markdown(
    """
    <style>
    body {
        background-image: url("C:/Users/sabar/PycharmProjects/price/Vertical-Apartment-Village-Singapore_9.jpg");  /* Set the background image URL */
        background-size: cover;  /* Cover the whole page */
        background-repeat: no-repeat;  /* Do not repeat the image */
        background-attachment: fixed;  /* Fixed background */
    }
    .title {
        color: #9933BD;
        font-size: 36px; 
        font-weight: bold;
    }
    .stButton > button {
        font-size: 20px;
        padding: 15px 30px;
        background-color: #4CAF50; 
        color: white; 
        border: none; 
        border-radius: 4px; 
    }
    .stButton > button:hover {
        background-color: #45a049; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title with the custom style
st.markdown('<h1 class="title">FLATS PRICE PREDICTION ( FLATS AT SINGAPOOR )</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.write("AWSOME FLATS IN SINGAPOOR")
    st.image("R.jpeg")
    st.image("hdb-flats.jpg")
    col1, col2 = st.columns(2)
    with col1:
        st.image('OIP.jpeg', width=300)
        st.image(Image.open("1142283.jpg"), width=255)
    with col2:
        st.image("469ed72ffd1881a89fa3c5943c399d27.jpg")

# User inputs
user_input = {}
colm3, colm4 = st.columns(2)

with colm3:
    year = st.slider("Which Year's price you want to know :", min_value=1990, max_value=2050)
    month = st.selectbox("Month", list(month_mapping.keys()))
    user_input['month'] = month_mapping[month]

with colm4:
    user_input['lease_commence_date'] = st.slider("Lease Commencement Year", min_value=1966, max_value=2023)
    town = st.selectbox("Town", all_towns)

user_input['floor_area_sqm'] = st.slider("Floor Area (sqm)", min_value=50, max_value=500)
user_input['remaining_lease'] = user_input['lease_commence_date'] + 99 - year
user_input['year'] = year

storey_range = st.selectbox("Storey Range", storey)
user_input['storey_range'] = convert_storey_range(storey_range)

flat_type = st.selectbox("Flat Type", list(flat_type_mapping.keys()))
user_input['flat_type'] = flat_type_mapping[flat_type]

for town_column in all_towns:
    user_input[f'town_{town_column}'] = 1 if town_column == town else 0

flat_model = st.selectbox("Flat Model", all_flat_models)
for flat_model_column in all_flat_models:
    user_input[f'flat_model_{flat_model_column}'] = 1 if flat_model_column == flat_model else 0

X_train_clms = ['floor_area_sqm', 'lease_commence_date', 'remaining_lease', 'year',
                'month', 'storey_range', 'flat_type'] + [f'town_{i}' for i in all_towns] + [f'flat_model_{i}' for i in all_flat_models]

# Prediction button
button = st.button("PREDICT")

if button:
    user_input_df = pd.DataFrame([user_input], columns=X_train_clms)
    loaded_model = load_model('random_forest_model.pkl')
    predicted_price = loaded_model.predict(user_input_df)
    st.write(f"### Predicted Resale Price: S$ {predicted_price[0]:.2f}")

