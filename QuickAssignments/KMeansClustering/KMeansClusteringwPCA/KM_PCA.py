
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score


import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


FILE   = "IRIS.csv"

df = pd.read_csv( FILE, encoding="ISO-8859-1" )
TARGET = "Species"
#print( df.head() )


X = df.copy()
X = X.drop( [TARGET], axis=1 )
X = X.drop( ["SepalWidth"], axis=1 )




varNames = X.columns

#print( X.head() )
#print( X.describe() )
#print( "\n\n")


##
##### TRANSFROM
##
theScaler = StandardScaler()
theScaler.fit( X )

X_TRN = theScaler.transform( X )
X_TRN = pd.DataFrame( X_TRN )
#print( X_TRN.head() )
#print( "\n\n")


pca = PCA()
pca.fit( X_TRN )
X_TRN = pca.transform( X_TRN )
X_TRN = pd.DataFrame( X_TRN )

varNames_trn = []
for i in range(X_TRN.shape[1]) :
    index = str(i+1)
    newName = "PC_" + index
    varNames_trn.append( newName )

X_TRN.columns = varNames_trn
print( X_TRN.head() )
print( "\n\n")

X_TRN = X_TRN.iloc[:,0:2]
print( X_TRN.head() )
print( "\n\n")



K_LIST = []
I_LIST = []
S_LIST = []
C_LIST = []

for K in range(3,12) :
    km = KMeans( n_clusters=K, random_state = 1 )
    km.fit( X_TRN )
    #Y = km.predict( X_TRN )
    K_LIST.append( K )
    I_LIST.append( km.inertia_ )
    S_LIST.append( silhouette_score(X_TRN,km.labels_) )
    C_LIST.append( calinski_harabaz_score(X_TRN,km.labels_) )


def drawElbow( K, SCORE, LABEL ) :
    plt.plot( K , SCORE, 'ro-', linewidth=2)
    plt.title(LABEL)
    plt.xlabel('Clusters')
    plt.ylabel('Score')
    plt.show()


drawElbow( K_LIST, I_LIST, "Inertia" )
drawElbow( K_LIST, S_LIST, "Silhouette" )
drawElbow( K_LIST, C_LIST, "Calinski" )





def clusterData( DATA, TRN_DATA, K, TARGET ) :
    print("\n\n\n")
    print("K = ",K)
    print("=======")
    km = KMeans( n_clusters=K, random_state = 1 )
    km.fit( TRN_DATA )
    Y = km.predict( TRN_DATA )
    DATA["CLUSTER"] = Y
    print( DATA.head() )

    G = DATA.groupby("CLUSTER")
    print( G.mean() )
    print("\n\n\n")
    print( G[ TARGET ].value_counts() )



clusterData( df, X_TRN, 3, TARGET )
#clusterData( df, X_TRN, 4, TARGET )
#clusterData( df, X_TRN, 5, TARGET )













