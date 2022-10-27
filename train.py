import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Custome LabelEncoding Class 
from src.utils import LabelEncoding
from src.model import DLRMModel


data = pd.read_csv('./data/adult.csv')

data.head(2)

all_columns = data.columns

num_cols = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

cat_cols = list(set(all_columns)-set(num_cols))

target_cols = ['income']

print (num_cols)
print (cat_cols)
print (target_cols)

num_data = data[num_cols]
cat_data = data[cat_cols]
target = data[target_cols]

target[target_cols] = (target[target_cols]=='>50K').astype(int)

scaler = MinMaxScaler()
num_data_scaled = scaler.fit_transform(num_data)

lbenc = LabelEncoding()

lbenc.fit(cat_data, cat_data.columns)

cat_enc = lbenc.transform(cat_data)

cat_enc.head(2)

y = target[target_cols].values

X = [num_data.values]

cat_x = []
for col in cat_enc:
    cat_x.append(cat_enc[col].values)

X.append(cat_x)

feature_dic = {}
for col in cat_enc:
    feature_dic[col] = cat_enc[col].nunique()

dlrm = DLRMModel(num_data, cat_enc, feature_dic)

dlrm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dlrm.fit(X, y, epochs=1, steps_per_epoch=1)
