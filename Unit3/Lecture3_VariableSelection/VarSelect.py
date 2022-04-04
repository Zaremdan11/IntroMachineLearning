

import pandas as pd
from sklearn.datasets import load_boston

#importing the necessary libraries
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt



boston = load_boston()
#print(boston.data.shape)         # for dataset dimension
#print(boston.feature_names)      # for feature names
#print(boston.target)             # for target variable
#print(boston.DESCR)              # for data description



bos = pd.DataFrame(boston.data, columns = boston.feature_names)
bos['Price'] = boston.target


X = bos.copy()
X = X.drop("Price", axis=1)       # feature matrix 
Y = bos[ 'Price' ]           # target feature



#print( X.head().T )
#print( Y.head() )




varNames = list( X.columns.values )
maxCols = X.shape[1]



sfs = SFS(LinearRegression(),
           k_features=( 5, 9 ),
           forward=True,
           floating=False,
           scoring = 'r2',
           cv=3
           )
sfs.fit(X.values, Y.values)


theFigure = plot_sfs(sfs.get_metric_dict(), kind=None )
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()






dfm = pd.DataFrame.from_dict( sfs.get_metric_dict()).T
dfm_names = dfm.columns.values
dfm = dfm[ ['feature_names', 'avg_score'] ]
print( dfm.head(13) )
print("before converting")
dt = dfm.dtypes
print( dt )
dfm.avg_score = dfm.avg_score.astype(float)
print("after converting")
dt = dfm.dtypes
print( dt )

maxIndex = dfm.avg_score.argmax()
print("argmax")
print( dfm.iloc[ maxIndex, ] )
print(" ................... ")



dfm_sort = dfm.sort_values(by='avg_score', ascending=False )
print( dfm_sort )
print(" ................... ")


theVars = dfm.iloc[ maxIndex, ]
theVars = theVars.feature_names
print( theVars )



theVarNames = []
for i in theVars :
    index = int(i)
    try :
        theName = varNames[ index ]
        theVarNames.append( theName )
    except :
        pass


for i in theVarNames :
    print(i)


W = X[ theVarNames ]
m = LinearRegression()
m.fit( W, Y )

coef_dict = {}
coef_dict["INTERCEPT"] = m.intercept_
for coef, feat in zip(m.coef_,theVarNames):
    coef_dict[feat] = coef

for i in coef_dict :
    print( i, " = ", coef_dict[i]  )





"""




#theFigure = plot_sfs(sfs.get_metric_dict(), kind='std_dev')











"""















