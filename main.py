import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()

print(dir(iris))

df = pd.DataFrame(iris.data, columns=iris.feature_names)


df['target'] = iris.target

#print(df)

x = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = SVC()
model.fit(X_train, y_train)


print(model.score(X_test, y_test))

