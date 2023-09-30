import numpy as np
import pandas as pd
data_o = pd.read_csv('creditcard.csv')
data_o.info()
data_o.describe()
data_o.dtypes
data_o.isnull().sum()
#data_o = data_o.drop(['Time'],axis = 1)


#SPLITTING THE DATASET
X = data_o.iloc[:, :-1]
y = data_o.iloc[:, -1]



#BALANCING THE DATASET USING SMOTE FOR DATASET
import imblearn
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X, y = sm.fit_sample(X, y) 

sum(y==1)     



#DETECTING AND REMOVING OUTLIERS USING ZSCORE
from scipy import stats
z = np.abs(stats.zscore(X))
threshold = 3
outliers = np.where(z > 3)
X = X[(z < 3).all(axis=1)]
y = y[(z < 3).all(axis = 1)]




#SCALING THE FEATURES
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
X = scalar.fit_transform(X)

#FEATURE SELECTION BY SELECTKBEST
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
good = SelectKBest(score_func = chi2, k = 10)
fit = good.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data_o.columns)
selection = pd.concat([dfcolumns,dfscores],axis=1)
selection.columns = ['names','scores']
selection.nlargest(30,'scores')
 

#DROPPING COLOUMNS IN DATA SET
df = pd.DataFrame(data=X, columns=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Amount'])

df = df.drop(['V15','V22','V23'],axis = 1)
X = df.copy()

#CORELATION USING HEATMAP
import seaborn as sns
import matplotlib.pyplot as plt
cormat = data_o.corr()
top_corr_features = cormat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(data_o[top_corr_features].corr(),annot = True)


#SPLITTING DATASET INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state = 0)


#FITTING LOGISTIC REGRESSION FOR TRAINIG SET
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train,y_train)

#PREDICTING THE TEST SET RESULTS
#y_pred = classifier.predict(X_test)


#MAKING THE CONFUSION MATRIX
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred)


#CHECKING THE ACCURACY
#from sklearn.metrics import r2_score
#accuracy = r2_score(y_test , y_pred)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.metrics import r2_score
accuracy = r2_score(y_test , y_pred)



print('hELLO')