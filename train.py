import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Custome LabelEncoding Class 
from src.utils import LabelEncoding
from src.model import DLRMModel

BATCH=512

### DATASET ###
# Load Adult data set to predict the income over 50k
data = pd.read_csv('./data/adult.csv')

# Set the numercal features and categorical features
all_columns = data.columns
num_cols = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = list(set(all_columns)-set(num_cols))
target_cols = ['income']

# Set Data
num_data = data[num_cols][:BATCH]
cat_data = data[cat_cols][:BATCH]
target = data[target_cols][:BATCH]
target[target_cols] = (target[target_cols]=='>50K').astype(int)

scaler = MinMaxScaler()
num_data_scaled = scaler.fit_transform(num_data)

lbenc = LabelEncoding()
lbenc.fit(cat_data, cat_data.columns)

cat_enc = lbenc.transform(cat_data)

### MODEL ###
# Prepare training datasets
y = target[target_cols].values
X = [num_data.values]

cat_x = []
for col in cat_enc:
    cat_x.append(cat_enc[col].values)

X.append(cat_x)

feature_dic = {}
for col in cat_enc:
    feature_dic[col] = cat_enc[col].nunique()

# Fitting model
dlrm = DLRMModel(num_data, cat_enc, feature_dic)
dlrm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dlrm.fit(X, y, epochs=1, steps_per_epoch=1)
