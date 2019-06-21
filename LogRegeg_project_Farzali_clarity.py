"""Created on Fri Mar 15 19:15:15 2019
diamonds.csv
@author: Izadi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
 os.chdir('D:\\mining code')  
os.getcwd()
# index_col=0 to remvove Unknowm columb after index column
df = pd.read_csv('diamonds.csv', index_col=0)
df_1 = dg.isnull()
dgf_2 = dg.isnull().sum()
df_2
# NO NULL at all
df.shape # (53940, 10) 

df.head()
df.shape
df.info()
df.describe()
#Getting nominal column=cut names
z = set(df.cut)
z
#Changing Nominal to ordinal 0-4 to make them  numerical
df.cut =df.cut.map({'Fair':0, 'Good':1, 'Ideal':2, 'Premium':3, 'Very Good':4})
#Getting nominal column=clarity names 
w = set(df.color)
w
#Changing Nominal to ordinal 0-7 to make them numerical
df.color = df.color.map({'D':0, 'E':1, 'F':2, 'G':3, 'H':4, 'I':5, 'J':6})
#We drop Nominal column clarity
X =df.drop(['clarity'], axis=1) 
type(X)
# We get y as column=clarity
y = df.clarity
max_accu = 0
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
estimator = LogisticRegression() #use regression model for regression proble
#import Normalisation package
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state=0)
#pd.Series(y_test).value_counts()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score 
cm = confusion_matrix(y_test,y_pred)
precision_score(y_test,y_pred, average='macro')
recall_score(y_test,y_pred, average='macro')
accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression
new_object_params = estimator.get_params(deep=False)
from sklearn.model_selection import ShuffleSplit
n_samples =df.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
predicted = cross_validate(log, X, y, cv=5)
cross_val_score(estimator, X, y,cv =5)


import seaborn as sns
import math
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
df['clarity'].value_counts() 
sns.countplot(df['clarity'])
df['color'].value_counts() 
sns.countplot(df['color'])
df['cut'].value_counts() 
sns.countplot(df['cut'])
plt.plot(df.carat,df.price)
plt.scatter(df.carat,df.price)
df['cut'].value_counts().plot(kind='bar')
df['clarity'].value_counts().plot(kind='bar')
df['color'].value_counts().plot(kind='bar')
df['cut'].value_counts().plot(kind='bar')
df['carat'].value_counts().plot(kind='bar')
plt.scatter(df.carat, df.price)

from statsmodels.graphics.mosaicplot import mosaic
plt.rcParams['font.size'] = 16.0
mosaic(df, ['cut', 'color'])
mosaic(df, ['cut', 'color', 'clarity'])
values = [21551, 13791, 12082, 4906, 1610]
labels = ['Ideal', 'Premium', 'Very Good', 'Good','Fair']
colors = ['b', 'g', 'r', 'c', 'm']
labels =labels 
plt.pie(values, colors=colors, labels= labels, counterclock=False, shadow=True)

df.corr(method='pearson')        # By default corr() is pearson
df.corr(method='spearman')
df.corr(method='kendall')

# from def to plt.show excute alltogether then correlation_matrix(ddef correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(df.corr('kendall'), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('crd Feature Correlation')
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=20)
    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
    
correlation_matrix(df)



















