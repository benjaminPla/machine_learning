from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("../data/one.csv", sep="\t")

#  student_dummies = pd.get_dummies(
#  data.student, drop_first=True, prefix="student", dtype=int
#  )
#  print(data, student_dummies)

#  data = pd.concat([data, student_dummies], axis=1)
data.student = data.student.map({"No": 0, "Yes": 1})

X = data[["student", "balance", "income"]]
y = data.default

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model = LogisticRegression(penalty=None)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(confusion_matrix(y_test, y_pred))
#  [[2404ok, 13wrong], [47ok, 36wrong]]
