# Data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


#Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
#Selects all the columns except the last one
x = dataset.iloc[:,:-1].values
#Selecting the last column now
y = dataset.iloc[:,-1].values

# Encoding the categorical varaibles now
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(), [3])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#Splitting the dataset into the Training Set and Test set
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state = 0)

#Training the multiple linear Regress model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test results now
y_pred = regressor.predict(X_test)
# Setting the precision to two varaibles
np.set_printoptions(precision=2)
#Displaying the two vecotrs now
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))