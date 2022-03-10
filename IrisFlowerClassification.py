# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 00:16:34 2022

@original_code_author: Srishti Sawla
@modified_code_author: Muhammed Cinsdikici

DataSet: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
Ref: https://medium.com/@srishtisawla/iris-flower-classification-fb6189de3fff

Data Set Descriptions:
----------------------
The dataset provided has 150 rows 
Dependent Variables         :   Sepal length.Sepal Width,
                                Petal length,Petal Width
Independent/Target Variable :   Class
Missing values              :   None
    
Attribute (Features) Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   
Labels (Outputs) of Classes:
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

path="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data=pd.read_csv(path,header=None,names=['sepal_length','sepal_width','petal_length','petal_width','class'])

# Data Set is splitted as %80 Training, %20 Testing
train,test = train_test_split(data,test_size=0.2,random_state=7)

print ("train :",train.shape, " test shape :", test.shape)
print ("Unique classes with count : ", pd.value_counts(train['class']))
print ("data description : ", train.describe())

#checking missing values
print ("Train Info : ",train.info())

######### EXPLORATORY DATA ANALYSIS ##########
print ("====>  Univariate analysis <======")

plt.title("Target Class Histogram")
plt.hist(train['class'])
plt.show()

print("Distributions of Sepal_length/Width and Petal Length/Width")
sns.displot(train['sepal_length'],kde=True,bins=40,color="blue")
plt.show()
sns.displot(train['sepal_width'],kde=True,bins=40,color="cyan")
plt.show()
sns.displot(train['petal_length'],kde=True,bins=40,color="red")
plt.show()
sns.displot(train['petal_width'],kde=True,bins=40,color="orange")
plt.show()

print ("====>  Bivariate analysis <======")
sns.boxplot(x='class',y='sepal_length',data=train,palette='OrRd')
plt.show()
sns.boxplot(x='class',y='sepal_width',data=train,palette='OrRd')
plt.show()
sns.boxplot(x='class',y='petal_length',data=train,palette='OrRd')
plt.show()
sns.boxplot(x='class',y='petal_width',data=train,palette='OrRd')
plt.show()
sns.heatmap(data.corr(),cmap="OrRd", linecolor='white', linewidths=1)
plt.show()
sns.pairplot(train, hue='class',palette='OrRd')
plt.show()

#### Converting categorical Class names to numeric Labels in Classic Way
# Using the index values...
# DONT FORGET THE CLASS COLUMN is STILL OBJECT TYPE ! 
trcp =train.copy()
mask =trcp["class"]=="Iris-versicolor"
trcp.loc[mask, 'class'] = 0

mask =trcp["class"]=="Iris-setosa"
trcp.loc[mask, 'class'] = 1

mask =trcp["class"]=="Iris-virginica"
trcp.loc[mask, 'class'] = 2

train = trcp
######
tstcp=test.copy()
mask =tstcp["class"]=="Iris-versicolor"
tstcp.loc[mask, 'class'] = 0

mask =tstcp["class"]=="Iris-setosa"
tstcp.loc[mask, 'class'] = 1

mask =tstcp["class"]=="Iris-virginica"
tstcp.loc[mask, 'class'] = 2
test=tstcp
######

# y: In the Train set, last column is class (numeric now) label
# X: In the Train set, first 4 columns are features (s_len, s_wid, p_len, p_wid)

X_train = train.iloc[:,:-1]
y_train = pd.to_numeric(train.iloc[:,-1])  #Now object types are converted to int64


X_test=test.iloc[:,:-1]
y_test=pd.to_numeric(test.iloc[:,-1])

"""
.iloc[] is 
    primarily integer position based (from 0 to length-1 of 
    the axis), but may also be used with a boolean array.
    ex: train.iloc[0]-> 0.raw informations as individual integers.
    ex: train.iloc[[0]]-> 0.raw info as integer array
        train[0:1] is same train.iloc[[0]]
    ex: train.iloc[i,j]-> the i.th row j.th column value of dataframe
    ex: train.iloc[:,j]-> the j.th column of dataframe
"""

# Spot Check Algorithms
models = []
#models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
scoring = 'accuracy'



results = []
names = []
for name, model in models:
    # TRAINING PHASE with k-Fold Cross Validation / TRAINING ACCURACY
	kfold = model_selection.KFold(n_splits=10, shuffle=False)
	cv_results = model_selection.cross_val_score(model, X_train,y_train,cv=kfold,scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg) 

for name, model in models:
	# TESTING PHASE / TEST ACCURACY
	print(name,"Testing Scores")
	model.fit(X_train,y_train)
	predictions = model.predict(X_test)
	print ("Accur: ",accuracy_score(y_test, predictions))
	print ("Confusion Matrix\n",confusion_matrix(y_test, predictions))
	print ("Classification Report\n",classification_report(y_test, predictions))

    



    
    




