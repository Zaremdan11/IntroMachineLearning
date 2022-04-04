

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")


FILE = "Steam_Logit_Data.csv"


df = pd.read_csv( FILE )

X = df.copy()
X = X.drop( "Y", axis=1 )
Y = df["Y"]


varNames = list( X.columns.values )
m = LogisticRegression()
m.fit(X,Y)




coef_dict = {}
coef_dict["INTERCEPT"] = m.intercept_[0]
for coef, feat in zip(m.coef_[0],varNames):
    coef_dict[feat] = coef

for i in coef_dict :
    print( i, " = ", coef_dict[i]  )



PRED_FLAG = m.predict( X )
print(" Predicted Flag Values ")
print( PRED_FLAG )
print(" --------- ")

PRED_PROB = m.predict_proba( X )
print(" Predicted Probabilities of 0 and 1")
print( PRED_PROB )
print(" --------- ")

P1 = PRED_PROB[:,1]
print(" Predicted Probability of 1")
print( P1 )
print(" --------- ")



fpr_train, tpr_train, threshold = metrics.roc_curve( Y, P1)
auc = metrics.auc(fpr_train, tpr_train)
print("AUC=",auc)

accuracy = metrics.accuracy_score( Y, PRED_FLAG )
print("ACCURACY=",accuracy)


