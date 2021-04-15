    import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv("../input/test.csv")

combine = [train_df, test_df]

print(train_df.columns.values)


train_df.head()
train_df.tail()

train_df.info()
print('-'*40)
test_df.info()

train_df.describe()

train_df.describe(include=['O'])


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)

train_df[['Sex','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
