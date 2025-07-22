# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report   

# Load your processed dataset
df = pd.read_csv('C:\\Users\\LENOVO\\Downloads\\PS_20174392719_1491204439457_log.csv.zip', compression='zip')  # already encoded & scaled if necessary

X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

with mlflow.start_run():

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log params, metrics, model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(clf, "model")

    print(f"Logged Model with accuracy: {acc}")
