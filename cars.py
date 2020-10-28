import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Path of the file to read
cars_file_path = "train-data.csv"
car_data = pd.read_csv(cars_file_path)

# Create target object and call it y
y = car_data.Price

# Create features and call it X
features = [
    'Name',
    'Location',
    'Year',
    'Kilometers_Driven',
    'Fuel_Type',
    'Transmission',
    'Owner_Type',
    'Mileage',
    'Engine',
    'Power',
    'Seats'
]
X = car_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify model
car_model = DecisionTreeRegressor(random_state=1)
# Fit model
car_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error (MAE)
val_predictions = car_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {0}".format(val_mae))

# Using best value for max_lead_nodes
car_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
car_model.fit(train_X, train_y)
val_predictions = car_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {0}".format(val_mae))

# Using a Random Forest
car_model = RandomForestRegressor(random_state=1)
car_model.fit(train_X, train_y)
val_predictions = car_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for Random Forest Model: {0}".format(val_mae))