from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("../data/three.csv", header=0)

#  print(df.info())
#  print(df.Time_taken.mean())
time_take_mean = df["Time_taken"].mean()
df["Time_taken"] = df["Time_taken"].fillna(time_take_mean)
#  print(df.info())

#  print(df.dtypes.loc[df.dtypes == "object"])
#  print(df["3D_available"].unique())
#  print(df["Genre"].unique())
df["3D_available"] = df["3D_available"].map({"NO": 0, "YES": 1})
#  df = pd.get_dummies(df, columns=["3D_available"], drop_first=True, dtype=int)
#  df.rename(columns={'3D_available_YES': '3D_available'}, inplace=True)
df = pd.get_dummies(df, columns=["Genre"], dtype=int)
#  print(df.info())

X = df.drop(columns=["Start_Tech_Oscar"])
y = df["Start_Tech_Oscar"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  X_train.to_csv("./X_train.csv", sep=",", header=True, index=False)
#  X_test.to_csv("./X_test.csv", sep=",", header=True, index=False)
#  y_train.to_csv("./y_train.csv", sep=",", header=True, index=False)
#  y_test.to_csv("./y_test.csv", sep=",", header=True, index=False)

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
#  print(model.feature_importances_)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
