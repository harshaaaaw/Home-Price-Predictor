from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle as pkl
import seaborn as sns
import numpy as np

# Load the dataset
data = pd.read_csv('train.csv')

# Encode categorical feature 'Location'
le = LabelEncoder()
data['Location'] = le.fit_transform(data['Location'])

# Log-transform the target variable 'Price' to stabilize variance
data['Price'] = np.log(data['Price'])

# Drop irrelevant features
x = data.drop(["id", "Price", "Lift Available", 'Clubhouse', "Maintenance Staff", "24x7 Security",
               "Children's Play Area", "Intercom", 'Swimming Pool', 'Gas Connection', "Landscaped Gardens"], axis=1)
y = data['Price']

# Handle outliers for 'Area'
q1 = x['Area'].quantile(0.25)
q3 = x['Area'].quantile(0.75)
iqr = q3 - q1
u = q3 + 1.5 * iqr
l = q1 - 1.5 * iqr
x['Area'] = np.clip(x['Area'], l, u)

# Handle outliers for 'Price'
q1 = y.quantile(0.25)
q3 = y.quantile(0.75)
iqr = q3 - q1
u = q3 + 1.5 * iqr
l = q1 - 1.5 * iqr
y = np.clip(y, l, u)

# Split the data into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)

# Initialize and train RandomForestRegressor model
rfc = RandomForestRegressor()
rfc.fit(x_train, y_train)

# Make predictions
y_pred = rfc.predict(x_test)

# Print the R-squared score for model evaluation
print(f"R-squared score: {r2_score(y_test, y_pred)}")

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pkl.dump(rfc, model_file)
