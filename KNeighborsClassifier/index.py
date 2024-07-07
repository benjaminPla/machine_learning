from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("../data/five.csv", sep=";")

X = df.drop(["Adulto", "Sex"], axis=1)
y = df["Adulto"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


knn = KNeighborsClassifier()
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
folds = StratifiedKFold(n_splits=10, shuffle=True)

grid = GridSearchCV(knn, param_grid, cv=folds, scoring="accuracy")
grid.fit(X_train, y_train)
y_pred_grid = grid.predict(X_test)
print(grid.best_estimator_)
print(grid.best_score_)
print(grid.best_params_)
print(classification_report(y_test, y_pred_grid))

random = RandomizedSearchCV(knn, param_grid, n_iter=20, cv=folds, scoring="accuracy")
random.fit(X_train, y_train)
y_pred_rand = random.predict(X_test)
print(random.best_estimator_)
print(random.best_score_)
print(random.best_params_)
print(classification_report(y_test, y_pred_rand))
