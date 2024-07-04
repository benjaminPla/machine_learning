from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

(X, y) = datasets.load_wine(return_X_y=True, as_frame=True)

model = DecisionTreeClassifier(max_depth=3)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X_train.columns,
    filled=True,
    label="none",
    impurity=False,
)
plt.savefig("decision_tree_plot.png")
plt.close()

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
