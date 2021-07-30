"""
File to ablate specified features when analyzing BM
Under the ABLATION hyper parameters, specify 0 to not ablate a certain parameter and 1 to do so
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
ABLATION = {"Region": 0,
            "Field Type": 0,
            "Quarter Type": 0,
            "Area Harvested": 0,
            "Rainfall of 3rd Month of Quarter": 0,
            "Rainfall of 2nd Month of Quarter": 0,
            "Rainfall of 1st Month of Quarter": 0,
            "Rainfall 1 Month Before Start of Quarter": 0,
            "Rainfall 2 Months Before Start of Quarter": 0,
            "Rainfall 3 Months Before Start of Quarter": 0,
            "SSTA El Nino 3.4 of 3rd Month of Quarter": 1,
            "SSTA El Nino 3.4 of 2nd Month of Quarter": 1,
            "SSTA El Nino 3.4 of 1st Month of Quarter": 1,
            "SSTA El Nino 3.4 1 Month Before Start of Quarter": 1,
            "SSTA El Nino 3.4 2 Months Before Start of Quarter": 1,
            "SSTA El Nino 3.4 3 Months Before Start of Quarter": 1}

# Load datasets into memory
dataset = pd.read_csv("../TESTING_DATASET.csv")

# Attain features name and label name
feature_names = dataset.columns[1:17]
label_name = dataset.columns[-1]

# Isolates the Labels (Y) and the Features (X)
labels = np.array(dataset.pop("Production"))
features = dataset.drop(["Quarters"], axis=1)

# Loads translation key into memory
translation = json.loads(open("../translation.json").read())

# Translates all non-numerical values into float64 values
features = features.replace(translation["Region"]) \
    .replace(translation["Field Type"]) \
    .replace(translation["Quarter Type"])

# Applies ablation properties
for key in ABLATION:
    if ABLATION[key] is 1:
        features[key].values[:] = 0

# Reshapes Labels before normalization
labels = labels.reshape(-1, 1)

# Initializes MinMax scaler for Labels and Features
scalers = [MinMaxScaler(feature_range=(0, 10))] * 2

# Normalizes Labels and Features
features[feature_names[3:]] = scalers[0].fit_transform(features[feature_names[3:]])
labels = scalers[1].fit_transform(labels)

# Loads model into memory
model = tf.keras.models.load_model("../BM")

# Inverses true and predicted values
actual = scalers[1].inverse_transform(labels)
predicted = scalers[1].inverse_transform(model.predict(features))

# Clear console
os.system("cls")

# Calculate R^2 ADJ
mean_actual = np.mean(actual)
r_squared = 1 - (np.sum(np.square(actual - predicted))/np.sum(np.square(actual - mean_actual)))
r_squared_adjusted = 1 - ((1 - r_squared)*(actual.shape[0] - 1))/(actual.shape[0] - len(ABLATION) - 1)

# Calculates MAE
mae = np.average(np.absolute(predicted - actual))

# Attain MSE
mse = model.evaluate(features, labels, batch_size=1024, verbose=0)

# Prints results
print("R2 ADJ: " + str(r_squared))
print("MAE: " + str(mae))
print("MSE: " + str(mse))

# Prints ablated features
print("\nAblated Features:")
i = 0
for key in ABLATION:
    i += 1
    if ABLATION[key] is 1:
        print(key + ": F" + str(i))