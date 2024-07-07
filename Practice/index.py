from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv("../data/seven.csv")
df = df.drop(["education"], axis=1)
#  print(df.isna().sum())
df = df.dropna()
#  columns_to_fill = [
#  "BMI",
#  "cigsPerDay",
#  "glucose",
#  "heartRate",
#  "totChol",
#  ]
#  for column in columns_to_fill:
#  mean_value = df[column].mean()
#  df[column] = df[column].fillna(mean_value)

x_columns = [
    "age",
    "BMI",
    "cigsPerDay",
    "diaBP",
    "glucose",
    "heartRate",
    "sysBP",
    "totChol",
]
X = df[x_columns]
y = df["TenYearCHD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

print("\n----------------------------------------------------\n")
print("DecisionTreeClassifier\n")
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
pred_df = model_dt.predict(X_test)
print(f"classification_report:\n{classification_report(y_test,pred_df)}\n")
print(f"confusion_matrix:\n{confusion_matrix(y_test,pred_df)}")
print("\n----------------------------------------------------\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n----------------------------------------------------\n")
print("LogisticRegression\n")
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)
pred_lr = model_lr.predict(X_test_scaled)
print(f"classification_report:\n{classification_report(y_test,pred_lr)}\n")
print(f"confusion_matrix:\n{confusion_matrix(y_test,pred_lr)}")
print("\n----------------------------------------------------\n")

knn = KNeighborsClassifier()
k_range = list(range(1, 51))
param_grid = dict(n_neighbors=k_range)
folds = StratifiedKFold(n_splits=10, shuffle=True)

print("\n----------------------------------------------------\n")
print("GridSearchCV\n")
model_grid = GridSearchCV(knn, param_grid, cv=folds, scoring="accuracy")
model_grid.fit(X_train, y_train)
pred_grid = model_grid.predict(X_test)
print(model_grid.best_params_)
print(f"classification_report:\n{classification_report(y_test,pred_grid)}\n")
print(f"confusion_matrix:\n{confusion_matrix(y_test,pred_grid)}")
print("\n----------------------------------------------------\n")

print("\n----------------------------------------------------\n")
print("RandomizedSearchCV\n")
model_rand = RandomizedSearchCV(
    knn, param_grid, n_iter=20, cv=folds, scoring="accuracy"
)
model_rand.fit(X_train_scaled, y_train)
pred_rand = model_rand.predict(X_test_scaled)
print(model_rand.best_params_)
print(f"classification_report:\n{classification_report(y_test,pred_rand)}\n")
print(f"confusion_matrix:\n{confusion_matrix(y_test,pred_rand)}")
print("\n----------------------------------------------------\n")

print("\n----------------------------------------------------\n")
print("KNeighborsClassifier\n")
model_knn = KNeighborsClassifier(n_neighbors=16)
model_knn.fit(X_train_scaled, y_train)
pred_knn = model_knn.predict(X_test_scaled)
print(f"classification_report:\n{classification_report(y_test,pred_knn)}\n")
print(f"confusion_matrix:\n{confusion_matrix(y_test,pred_knn)}")
print("\n----------------------------------------------------\n")

print("\n----------------------------------------------------\n")
print("XGBClassifier\n")
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)
print(f"classification_report:\n{classification_report(y_test,pred_xgb)}\n")
print(f"confusion_matrix:\n{confusion_matrix(y_test,pred_xgb)}")
print("\n----------------------------------------------------\n")
