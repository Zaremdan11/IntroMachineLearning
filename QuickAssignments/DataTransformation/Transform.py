
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler




pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)



FILE   = "IRIS.csv"

df = pd.read_csv( FILE, encoding="ISO-8859-1" )
TARGET = "Species"


X = df.copy()
X = X.drop( [TARGET], axis=1 )
varNames = X.columns

print( X.head() )
print( X.describe() )
print( "\n\n")




### MIN MAX SCALER
print(" NORMALIZING THE DATA \n\n\n")
theScaler = MinMaxScaler()
theScaler.fit( X )

X_MINMAX = theScaler.transform( X )
X_MINMAX = pd.DataFrame( X_MINMAX )
print( X_MINMAX.head() )
print( "\n\n")

varNames_minmax = []
for i in varNames :
    newName = "nor_" + i
    varNames_minmax.append( newName )
print( varNames_minmax )
print( "\n\n")

X_MINMAX.columns = varNames_minmax
print( X_MINMAX.head() )
print( "\n\n")

print( X_MINMAX.describe() )
print( "\n\n")



X_MINMAX[ "TARGET" ] = df.Species
print( X_MINMAX.head() )
print( "\n\n")

X_NEW = pd.concat([ X , X_MINMAX ], axis=1 )
print( X_NEW.head() )
print( "\n\n")


X_SMALL = X.iloc[ 0:3, ]
print( X_SMALL ) 
X_SMALL_MINMAX = theScaler.transform( X_SMALL )
X_SMALL_MINMAX = pd.DataFrame( X_SMALL_MINMAX )
X_SMALL_MINMAX.columns = varNames_minmax
print( X_SMALL_MINMAX.head() )
print( "\n\n")


X_TEST = X_NEW[ ["SepalLength", "nor_SepalLength" ] ]
print( X_TEST.head() ) 
print( "\n\n")
print( X_TEST["SepalLength"].describe() )
print( "\n\n")
TEMP = ( X_TEST["SepalLength"] - 4.3 ) / ( 7.9 - 4.3 )
#TEMP = ( X_TEST["SepalLength"] - 4.3 ) / ( 10000 - 4.3 )    # this is what happens when you have outliers.
X_TEST = X_TEST.assign( calc_SepalLength = TEMP.values )
print( X_TEST.head() ) 
print( "\n\n")








# STANDARD SCALER

print(" STANDARDIZING THE DATA \n\n\n")

theScaler = StandardScaler()
theScaler.fit( X )

Y_STD = theScaler.transform( X )
Y_STD = pd.DataFrame( Y_STD )
print( Y_STD.head() )
print( "\n\n")

varNames_std = []
for i in varNames :
    newName = "std_" + i
    varNames_std.append( newName )

Y_STD.columns = varNames_std
print( Y_STD.head() )
print( "\n\n")

print( Y_STD.describe() )
print( "\n\n")


Y_STD[ "TARGET" ] = df.Species
print( Y_STD.head() )
print( "\n\n")


Y_NEW = pd.concat([ X , Y_STD ], axis=1 )
print(Y_NEW.head() )
print( "\n\n")


Y_SMALL = X.iloc[ 0:3, ]
Y_SMALL_STD = theScaler.transform( Y_SMALL )
Y_SMALL_STD = pd.DataFrame( Y_SMALL_STD )
Y_SMALL_STD.columns = varNames_std
print( Y_SMALL_STD.head() )
print( "\n\n")


Y_TEST = Y_NEW[ ["SepalLength", "std_SepalLength" ] ]
print( Y_TEST.head() ) 
print( "\n\n")
print( Y_TEST["SepalLength"].describe() )
print( "\n\n")
TEMP = ( Y_TEST["SepalLength"] - 5.843333 ) / 0.828066
Y_TEST = Y_TEST.assign( calc_SepalLength = TEMP.values )
print( Y_TEST.head() ) 
print( "\n\n")











