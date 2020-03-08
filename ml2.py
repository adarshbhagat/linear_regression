import pandas as pd
import numpy as np#computing library
import quandl,math
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate,train_test_split
from sklearn.linear_model import LinearRegression
#Data Scaling or Standardization:
#It is a step of Data Pre Processing which is applied to independent variables
#or features of data. It basically helps to normalise the data
#within a particular range.
#Sometimes, it also helps in speeding up the calculations in an algorithm.

#cross_validaton helps in training and testing data also separates the data
#it can save our time

#With supervised learning, you have features and labels.
#The features are the descriptive attributes,
#and the label is what you're attempting to predict or forecast. ...
#Thus, for training the machine learning classifier,
#the features are customer attributes,
#the label is the premium associated with those attributes.

df = quandl.get("BSE/SPBSS5IP")
df['diff_oc']=(df['Close']-df['Open'])
df['diff_hl']=df['High']-df['Low']
df = df[['Open','Close','diff_hl','diff_oc']]
forecast_col='Close'
df['label']=df[forecast_col].shift(-1)
df.dropna(inplace=True)#drop rows with NaN values

x = np.array(df.drop(['label'],1))#assign every value to x which is not in label
y = np.array(df['label'])#assign label data to y

x=preprocessing.scale(x)
#scaling x. it will remove the sparsity in data
#scaling is required only when data is in skewed form, after scaling data will
# be in small range. it will improve the algorithm by reducing time for
#calculation.

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
#test_size is the amount of data to be used for test and train

clf = LinearRegression()
clf.fit(x_train,y_train)
accuracy = clf.score(x_test,y_test)
print(accuracy)
