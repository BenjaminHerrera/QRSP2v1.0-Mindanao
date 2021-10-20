# QRSP2v1.0-Mindanao
QRSP2v1.0-Mindanao (Quarterly and Regional System Palay Predictor v1.0 Mindanao) is a deep learning model that
predicts the following quarter's rice production level for the regions of Mindanao.
The required inputs and following output for QRSP2v1.0-Mindanao are described in this
[dataset repository](https://www.kaggle.com/benjlh/qrsp2v10mindanao-dataset). 

# Files
`./M_ReLU_50-30_BN.py` : Python file that made QRSP2v1.0-Mindanao

`./translation.py` : A translation key that is used to translate non-numerical values into numerical

`./eval/ablation.py` : File used to conduct the ablation study

`./eval/input_error.py` : File used to conduct error induct input study

`./BM/` : Folder that contains model files for BM

**NOTE:**
For `./eval/ablation.py` and `./eval/input_error.py`, the files can only be run on Windows because of `line 75`. 
For Linux users, change `os.system("cls")` to `os.system("clear")`

# Model Loading
To load the model, utilize tensorflow's `tensorflow.keras.models.load_model()` method
Below is an example
```python
import tensorflow as tf

model = tf.keras.models.load_model("../BM")
```

# Model Predicting
To conduct a prediction with QRSP2v1.0-Mindanao, you must first setup the `TESTING` dataset.
To accomplish this, follow these steps:
- Load dataset into memory
- Pop `Productions` column into another variable called `labels`
- Drop the `Quarters` column; the resulting dataframe will be assinged to a variable called `features`
- Load translation key
- Translate non-numerical values with translation key
- Reshape `features`
- Initialize two MinMax Scalers with a features range of 0 to 10
- Normalize `labels` and `features` with you scalers
- Load model
- Create a list that will hold your inputs; these inputs are the features that are detailed in the paper (must be in order)
- Transform your inputs with the scaler used to normalize `features`
- Pass your transformed inputs to the model and assign results to a variable called `results`
- Reverse the normalization of `results` with the scaler used to normalize `labels`

The following is an example of how to conduct a prediciton:
```python
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import json

# Load datasets into memory
dataset = pd.read_csv("./TESTING_DATASET.csv")

# Attain features name and label name
feature_names = dataset.columns[1:17]
label_name = dataset.columns[-1]

# Isolates the Labels (Y) and the Features (X)
labels = np.array(dataset.pop("Production"))
features = dataset.drop(["Quarters"], axis=1)

# Loads translation key into memory
translation = json.loads(open("./translation.json").read())

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
model = tf.keras.models.load_model("./BM")

# Your inputs
inputs = [<YOUR INPUTS>]

# Transform your inputs
inputs = label_scaler.transform(inputs)

# Inverses predicted values
predicted = label_scaler.inverse_transform(model.predict(features))

# Print predicted value
print("I predicted: " + str(predicted) + " tons")
```

# Links
**Dataset Repository:** https://www.kaggle.com/benjlh/qrsp2v10mindanao-dataset
