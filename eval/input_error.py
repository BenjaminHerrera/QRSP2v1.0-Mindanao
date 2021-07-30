"""
File to induce error into specified features when analyzing BM
Under the ERRORS hyper parameters, specify a percentage (e.g. 10 for 10% or -69 for 69%) 
to observe change in BM's performance
"""

# Imports
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os

# Prevents CUBLAS_STATUS_ALLOC_FAILED errors
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Hyper Parameters
ERRORS = {"Rainfall of 3rd Month of Quarter": 0,
          "Rainfall of 2nd Month of Quarter": 0,
          "Rainfall of 1st Month of Quarter": 0,
          "SSTA El Nino 3.4 of 3rd Month of Quarter": -15,
          "SSTA El Nino 3.4 of 2nd Month of Quarter": -12.5,
          "SSTA El Nino 3.4 of 1st Month of Quarter": -10}

# Load datasets into memory
dataset = pd.read_csv("../TESTING_DATASET.csv")

# Attain features name and label name
feature_names = dataset.columns[1:17]
label_name = dataset.columns[-1]

# Isolates the Labels (Y) and the Features (X)
labels = np.array(dataset.pop("Production"))
features = dataset.drop(["Quarters"], axis=1)
features_copy = features

# Loads translation key into memory
translation = json.loads(open("../translation.json").read())

# Translates all non-numerical values into float64 values
features = features.replace(translation["Region"]) \
    .replace(translation["Field Type"]) \
    .replace(translation["Quarter Type"])

# Reshapes Labels before normalization
labels = labels.reshape(-1, 1)

# Initializes MinMax scalers for Labels and Features
features_scaler = MinMaxScaler(feature_range=(0, 10))
label_scaler = MinMaxScaler(feature_range=(0, 10))

# Normalizes Labels and Features
features_copy[feature_names[3:]] = features_scaler.fit_transform(features_copy[feature_names[3:]])
labels = label_scaler.fit_transform(labels)

# Loads model into memory
model = tf.keras.models.load_model("../BM")

# Induce errors to specified features
for key in ERRORS:
    features[key].values[:] = features[key].values[:] * (1 + (ERRORS[key] / 100))

# Scales error induced features
features[feature_names[3:]] = features_scaler.transform(features[feature_names[3:]])

# Inverses true and predicted values
actual = label_scaler.inverse_transform(labels)
predicted = label_scaler.inverse_transform(model.predict(features))

# Clear console
os.system("cls")

# Calculate R^2 ADJ
mean_actual = np.mean(actual)
r_squared = 1 - (np.sum(np.square(actual - predicted))/np.sum(np.square(actual - mean_actual)))
r_squared_adjusted = 1 - ((1 - r_squared)*(actual.shape[0] - 1))/(actual.shape[0] - len(ERRORS) - 1)

# Calculates MAE
mae = np.average(np.absolute(predicted - actual))

# Attain MSE
mse = model.evaluate(features, labels, batch_size=1024, verbose=0)

# Prints results
print("R2 ADJ: " + str(r_squared))
print("MAE: " + str(mae))
print("MSE: " + str(mse))

# Prints errors induced into features
print("\nError Induced to Features:")
i = 0
for key in ERRORS:
    print(key + ": " + str(ERRORS[key]) + "%")