import pandas as pd
import joblib
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
dataset=fetch_california_housing()
feature_names = dataset['feature_names']
print("Feature names: {}\n".format(feature_names))
print(dataset.data)
data_original = (dataset.data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target,test_size = 0.40,random_state = 42)
model= LinearRegression().fit(X_train,y_train)
model_file=open('model.pkl','wb')
pickle.dump(model,model_file)


