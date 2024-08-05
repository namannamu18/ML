import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

classifier.fit(X_train, y_train)

new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = classifier.predict(new_sample)
print(f"Predicted class for the new sample: {iris.target_names[prediction[0]]}")

plt.figure(figsize=(20, 10))
tree.plot_tree(classifier, filled=True, feature_names=iris.feature_names, 
class_names=iris.target_names)
plt.show()
