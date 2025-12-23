from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

dt = tree.DecisionTreeClassifier()
data = pd.read_csv(r"D:\Internship\Pandas\fruit.csv")

columns = ['Color', 'Diameter', 'Fruit']

fig, ax = plt.subplots()
ay = ax.twiny()
colors = np.random.randint(10,100, data['Color'].size)
ax.scatter(data['Color'], data['Fruit'], c = colors, cmap = 'Pastel1')
ay.scatter(data['Diameter'], data['Fruit'], c = colors, cmap = 'twilight')
plt.show()

from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder()
enc = one_hot.fit_transform(data[["Color"]])
data[one_hot.categories_[0]] = enc.toarray()

X = data
X = X.drop(columns=['Color', 'Fruit'])
X = X.values
Y = data["Fruit"].values
Y = Y.reshape(Y.size, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = Y, test_size = 0.3, random_state = 0)

dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(dt.predict_proba(X_test))

plt.figure()
tree.plot_tree(dt, feature_names = data.columns, class_names = data['Fruit'].unique(), filled = True, rounded = True)
plt.show()
