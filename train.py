import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Custome LabelEncoding Class 
from src.utils import LabelEncoding
from src.model import DLRMModel

BATCH=512

# data preparation
data = pd.read_csv('./data/adult.csv')

all_columns = data.columns

num_cols = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = list(set(all_columns)-set(num_cols))
target_cols = ['income']

num_data = data[num_cols]
cat_data = data[cat_cols]
target = data[target_cols]

print(num_data[:BATCH])
print(cat_data[:BATCH])
print(target[:BATCH])

target[target_cols] = (target[target_cols]=='>50K').astype(int)

scaler = MinMaxScaler()
num_data_scaled = scaler.fit_transform(num_data)

lbenc = LabelEncoding()
lbenc.fit(cat_data, cat_data.columns)

cat_enc = lbenc.transform(cat_data)

y = target[target_cols].values
X = [num_data.values]

cat_x = []
for col in cat_enc:
    cat_x.append(cat_enc[col].values)

X.append(cat_x)

feature_dic = {}
for col in cat_enc:
    feature_dic[col] = cat_enc[col].nunique()

# generate model
dlrm = DLRMModel(num_data, cat_enc, feature_dic)
# compile model
dlrm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# train
dlrm.fit(X, y, epochs=1, steps_per_epoch=1)
