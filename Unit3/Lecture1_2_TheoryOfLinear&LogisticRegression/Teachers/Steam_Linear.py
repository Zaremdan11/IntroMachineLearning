

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics



FILE = "Steam_Linear_Data.csv"


df = pd.read_csv( FILE )

X = df.copy()
X = X.drop( "Y", axis=1 )
Y = df["Y"]


varNames = list( X.columns.values )
m = LinearRegression()
m.fit(X,Y)


coef_dict = {}
coef_dict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_,varNames):
    coef_dict[feat] = coef

for i in coef_dict :
    print( i, " = ", coef_dict[i]  )


PRED = m.predict( X )
print(" Predicted Values ")
print( PRED )
print(" --------- ")


r2 = metrics.r2_score( Y, PRED )
print( "r2=", r2 )

mse = metrics.mean_squared_error(Y,PRED)
print( "MSE=", mse )



