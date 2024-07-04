from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("../data/two.csv")

train_cols = [
    "average_montly_hours",
    "last_evaluation",
    "number_project",
    "promotion_last_5years",
    "satisfaction_level",
    "time_spend_company",
    "Work_accident",
]
X = df[train_cols]
y = df.left
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(C=1e10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print(recall_score(y_test, y_pred))
specificity = TN / (TN + FP)
print(specificity)
print(f1_score(y_test, y_pred))
