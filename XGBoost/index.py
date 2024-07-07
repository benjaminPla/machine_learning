from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv("../data/six.csv")
deletable_columns = [
    "Cloud3pm",
    "Cloud9am",
    "Date",
    "Evaporation",
    "Location",
    "RainToday",
    "Sunshine",
    "WindDir3pm",
    "WindDir9am",
    "WindGustDir",
]
df = df.drop(columns=deletable_columns)
df = df.dropna()
df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})

X = df[["MaxTemp", "Humidity3pm"]]
y = df["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
print(accuracy_score(y_test, pred_xgb))
print(classification_report(y_test, pred_xgb))
print(confusion_matrix(y_test, pred_xgb))
