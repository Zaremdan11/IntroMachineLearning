
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


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
print( "\n\n")

theScaler = StandardScaler()
theScaler.fit( X )

X_STD = theScaler.transform( X )


MAX_N = X_STD.shape[1]
pca = PCA(n_components=MAX_N)
pca.fit( X_STD )



ev = pca.explained_variance_
print("Eigen Values")
print(ev)
print("\n\n")


varPCT = []
totPCT = []
total = 0
for i in ev:
    total = total + i
    VAR = int( i / len(ev) * 100)
    PCT = int( total / len(ev) * 100)
    varPCT.append(VAR)
    totPCT.append( PCT )
    print( round(i,2), "variation=", VAR,"%"," total=", PCT,"%")


PC_NUM = np.arange( MAX_N ) + 1
plt.plot( PC_NUM , ev, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

PC_NUM = np.arange( MAX_N ) + 1
plt.plot( PC_NUM , varPCT, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()


PC_NUM = np.arange( MAX_N ) + 1
plt.plot( PC_NUM , totPCT, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Total Variance Explained')
plt.show()




dfc = pd.DataFrame( pca.components_ )
dfc.columns = list( X.columns)
print( dfc )

pca = PCA(n_components=MAX_N)
pca.fit( X_STD )


X_PCA = pca.transform( X_STD )
X_PCA = pd.DataFrame( X_PCA )
X_PCA = X_PCA.iloc[:,0:2]


colNames = X_PCA.columns
pcaNames = []
for i in colNames :
    index = int(i) + 1
    theName = "PC_" + str(index)
    pcaNames.append( theName )
    
X_PCA.columns = pcaNames

print( X_PCA.head() )
print("\n\n")


print( df.head() )
print("\n\n")

X_PCA["TARGET"] = df.Species
print( X_PCA.head() )
print("\n\n")




for Name, Group in X_PCA.groupby("TARGET"):
    print( Group.head() )
    print("\n")
    

for Name, Group in X_PCA.groupby("TARGET"):
    plt.scatter(Group.PC_1, Group.PC_2, label=Name)
plt.xlabel("PC_1")
plt.ylabel("PC_2")
plt.legend()
plt.show()




