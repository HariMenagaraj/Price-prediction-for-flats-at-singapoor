import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
singapoorFlat = pd.read_csv("C:/Users/sabar/OneDrive/Desktop/singapoor flats 1990-2024.csv")
singapoorFlat.drop(columns=['Unnamed: 0'], inplace=True)

# Extract year and month, calculate remaining lease
singapoorFlat['month'] = pd.to_numeric(singapoorFlat['month'].str.split('-').str[1])
singapoorFlat['year'] = pd.to_numeric(singapoorFlat['month'].str.split('-').str[0])
singapoorFlat['remaining_lease'] = singapoorFlat['lease_commence_date'] + 99 - singapoorFlat['year']

# Clean flat_type and flat_model
singapoorFlat['flat_type'] = singapoorFlat['flat_type'].str.replace('MULTI-GENERATION', 'MULTI GENERATION')
singapoorFlat['flat_model'] = singapoorFlat['flat_model'].str.title()

# Calculate storey range average
singapoorFlat['storey_range'] = singapoorFlat['storey_range'].apply(lambda x: (int(x.split(' TO ')[0]) + int(x.split(' TO ')[1])) / 2)

# Encode categorical variables
flat_type_mapping = {'1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4, '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI GENERATION': 7}
singapoorFlat['flat_type'] = singapoorFlat['flat_type'].map(flat_type_mapping)

# Drop unnecessary columns
singapoorFlat.drop(columns=['street_name', 'block'], inplace=True)

# One-hot encode town and flat_model
singapoorFlat_dum = pd.get_dummies(singapoorFlat, columns=['town', 'flat_model'])

# Plotting various visualizations
plt.figure(figsize=(15, 6))
sns.lineplot(x='year', y='resale_price', data=singapoorFlat).set(title='Resale Price Over Time', xlabel='Year', ylabel='Resale Price')
plt.xticks(rotation=75)
plt.show()

plt.figure(figsize=(15, 6))
sns.lineplot(x='remaining_lease', y='resale_price', data=singapoorFlat).set(title='Price based on remaining year', xlabel='Remaining Lease', ylabel='Resale Price')
plt.xticks(rotation=75)
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(x='flat_model', y='resale_price', data=singapoorFlat).set(title='Price based on flat model', xlabel='Flat Model', ylabel='Resale Price')
plt.xticks(rotation=75)
plt.show()

plt.figure(figsize=(20, 6))
sns.lineplot(x='floor_area_sqm', y='resale_price', data=singapoorFlat).set(title='Price based on floor area sqm', xlabel='Floor Area (sqm)', ylabel='Resale Price')
plt.xticks(rotation=75)
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(x='town', y='resale_price', data=singapoorFlat).set(title='Price based on town', xlabel='Town', ylabel='Resale Price')
plt.xticks(rotation=75)
plt.show()

plt.figure(figsize=(15, 8))
sns.barplot(x='flat_type', y='resale_price', data=singapoorFlat).set(title='Price based on flat type', xlabel='Flat Type', ylabel='Resale Price')
plt.xticks(rotation=75)
plt.show()

sns.scatterplot(x='resale_price', y='floor_area_sqm', hue='flat_type', data=singapoorFlat).set(title='Price based on floor area sqm', xlabel='Floor Area (sqm)', ylabel='Resale Price')
plt.show()

sns.pairplot(singapoorFlat, hue='resale_price')
plt.show()

numerical_features = singapoorFlat.select_dtypes(include=['number'])
sns.heatmap(numerical_features.corr(), annot=True)
plt.show()

# Train/Test split
X = singapoorFlat_dum.drop("resale_price", axis=1)
y = singapoorFlat_dum["resale_price"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=82)

# Model training
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(f"{name} Results:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_val, pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_val, pred)}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_val, pred))}")
    print(f"R2 Score: {r2_score(y_val, pred)}\n")

# Save the best model
best_model = models['Random Forest']
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Define a function to get user input
def get_user_input():
    user_input = {
        'month': int(input("Enter the month: ")),
        'floor_area_sqm': float(input("Enter the floor area in square meters: ")),
        'lease_commence_date': int(input("Enter the lease commencement year: ")),
        'year': int(input("Enter the year: "))
    }
    user_input['remaining_lease'] = user_input['lease_commence_date'] + 99 - user_input['year']
    user_input['storey_range'] = float(input("Enter the storey range: "))
    user_input['flat_type'] = int(input("Enter the flat type (e.g., 1, 2, 3, etc.): "))
    town = input("Enter the town (e.g., 'BEDOK', 'TAMPINES', etc.): ")
    flat_model = input("Enter the flat model (e.g., 'Improved', 'New Generation', etc.): ")

    all_town_columns = [f'town_{t}' for t in ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH',
        'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
        'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'LIM CHU KANG', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL',
        'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN']]
    for town_column in all_town_columns:
        user_input[town_column] = 1 if town_column == f'town_{town}' else 0

    all_flat_model_columns = [f'flat_model_{fm}' for fm in ['2-Room', '3Gen', 'Adjoined Flat', 'Apartment', 'Dbss', 'Improved',
        'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation',
        'Premium Apartment', 'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1', 'Type S2']]
    for flat_model_column in all_flat_model_columns:
        user_input[flat_model_column] = 1 if flat_model_column == f'flat_model_{flat_model}' else 0

    return user_input

# Get user input
user_input_data = get_user_input()

# Create a DataFrame from user input
user_input_df = pd.DataFrame([user_input_data], columns=X_train.columns)

# Load the model and make predictions
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
predicted_price = loaded_model.predict(user_input_df)
print("Predicted Resale Price:", predicted_price[0])
