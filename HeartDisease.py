# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#import os
#print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

dataset = pd.read_csv("../input/heart.csv")
#print (dataset.columns)

#for col in data.columns:
#    print (data.groupby([col, 'target']).describe())

X = dataset.loc[:, ['sex', 'cp', 'exang', 'slope', 'ca', 'thal', 'restecg', 'age']]
Y = dataset.loc[:, ['target']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=0)
model.fit(X_train, Y_train.values.ravel())
print(model.get_params())

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
print ("Accuracy:", accuracy_score(Y_test, predictions))
print ("Confusion Matrix:")
print (confusion_matrix(Y_test, predictions))