# Imports
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import time

# Assigns a name to the model
NAME = "M_ReLU_50-30_BN_{}".format(int(time.time()))

# Prevents CUBLAS_STATUS_ALLOC_FAILED errors
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.80))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Creates tensorboard callback
tensorboard = TensorBoard(log_dir='./logs')

# Initializes reccurent checkpoint callback
# Info:
#   Frequency: 100
reccurent_checkpoint_path = "./checkpoints/reccurent/cp-{epoch:03d}"
reccurent_checkpoint_callback = keras.callbacks.ModelCheckpoint(reccurent_checkpoint_path, period=100)

# Initializes best model checkpoint callback
best_model_checkpoint_path = "./checkpoints/best/cp-{epoch:03d}-{val_loss:.5f}"
best_model_checkpoint_callback = keras.callbacks.ModelCheckpoint(best_model_checkpoint_path, save_best_only=True)

# Load datasets into memory
TRAINING_dataset = pd.read_csv("TRAINING_DATASET.csv")
VALIDATION_dataset = pd.read_csv("VALIDATION_DATASET.csv")

# Isolates the Labels (Y) and the Features (X)
TRAINING_labels = np.array(TRAINING_dataset.pop("Production"))
TRAINING_features = TRAINING_dataset.drop(["Quarters"], axis=1)
VALIDATION_labels = np.array(VALIDATION_dataset.pop("Production"))
VALIDATION_features = VALIDATION_dataset.drop(["Quarters"], axis=1)

# Loads translation key into memory
translation = json.loads(open("translation.json").read())

# Translates all non-numerical values into float64 values
TRAINING_features = TRAINING_features.replace(translation["Region"])
TRAINING_features = TRAINING_features.replace(translation["Field Type"])
TRAINING_features = TRAINING_features.replace(translation["Quarter Type"])
VALIDATION_features = VALIDATION_features.replace(translation["Region"])
VALIDATION_features = VALIDATION_features.replace(translation["Field Type"])
VALIDATION_features = VALIDATION_features.replace(translation["Quarter Type"])

# Initializes MinMax scaler for Labels and Features
TRAINING_features_scaling = MinMaxScaler(feature_range=(0, 10))
TRAINING_labels_scaling = MinMaxScaler(feature_range=(0, 10))
VALIDATION_features_scaling = MinMaxScaler(feature_range=(0, 10))
VALIDATION_labels_scaling = MinMaxScaler(feature_range=(0, 10))

# Reshapes Labels before normalization
TRAINING_labels = TRAINING_labels.reshape(-1, 1)
VALIDATION_labels = VALIDATION_labels.reshape(-1, 1)

# Normalizes Labels and Features
features_columns = ["Area Harvested", "Rainfall of 3rd Month of Quarter", "Rainfall of 2nd Month of Quarter", "Rainfall of 1st Month of Quarter", "Rainfall 1 Month Before Start of Quarter", "Rainfall 2 Months Before Start of Quarter", "Rainfall 3 Months Before Start of Quarter", "SSTA El Nino 3.4 of 3rd Month of Quarter", "SSTA El Nino 3.4 of 2nd Month of Quarter", "SSTA El Nino 3.4 of 1st Month of Quarter", "SSTA El Nino 3.4 1 Month Before Start of Quarter", "SSTA El Nino 3.4 2 Months Before Start of Quarter", "SSTA El Nino 3.4 3 Months Before Start of Quarter"]
TRAINING_features[features_columns] = TRAINING_features_scaling.fit_transform(TRAINING_features[features_columns])
TRAINING_labels = TRAINING_labels_scaling.fit_transform(TRAINING_labels)
VALIDATION_features[features_columns] = VALIDATION_features_scaling.fit_transform(VALIDATION_features[features_columns])
VALIDATION_labels = VALIDATION_labels_scaling.fit_transform(VALIDATION_labels)

# Model Structure:
#   5 Input Nodes:
#       1. Region Name (Translated)
#       2. Field Type (Translated)
#       3. Area Harvested
#       4. Rainfall (6 Month Span)
#       5. ENSO (6 Month Span)
#   2 Hidden Layers:
#       1. 50 ReLU
#           - L2 WEIGHT Regularizer (0.01)
#       2. Dropout (0.25)
#       3. Batch Normalization
#       4. 30 ReLU
#           - L2 WEIGHT Regularizer (0.01)
#       5. Dropout (0.25)
#       6. Batch Normalization
#   1 Output Layer:
#       1. 1 Linear
model = keras.Sequential([
    keras.layers.Dense(50, input_shape=(16,), activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(30, activation='relu', kernel_regularizer=l2(0.01)),
    keras.layers.Dropout(0.25),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1)
])

# Training continuation
# Comment previous model declaration and un-comment the line below to load previous session
# Be sure to specify the right directory
# model = keras.models.load_model("../../checkpoints/D2/50-30/model_DMinMaxScaler-0-10-ALL_ReLU-50-30_WR-L2-0.01_D-0.25_VS-0.2_BS-16_1613466208/cp-128600")

# Displays model structure
print(model.summary())

# Model compilation
# Info:
#   Optimizer: ADAM
#   Learning Rate: 0.0001
#   Loss Function: Mean Squared Error
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Model training
# Info:
#   Batch Size: 32
#   Epochs: 40000
#   Validation Percentage: 20%
model.fit(x=TRAINING_features, y=TRAINING_labels, batch_size=1024, epochs=1000000, validation_data=(VALIDATION_features, VALIDATION_labels), callbacks=[tensorboard, best_model_checkpoint_callback, reccurent_checkpoint_callback])