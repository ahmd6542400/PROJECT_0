import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load the Boston Housing dataset
data = pd.read_csv('score.csv')

# Split the data into features (X) and target variable (y)
X = data.drop('Scores', axis=1)
y = data['Scores']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)

# Saving model using pickle
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))