import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

from src.preprocessing.feature_engineer import ROOT_DIR, PROCESSED_FILE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib


# Load dataset
df = pd.read_csv(PROCESSED_FILE)

# Split features and target
X = df.drop(columns='RainTomorrow')
y_class = df['RainTomorrow']

# Train-test split with stratification
x_train, x_test, y_class_train, y_class_test = train_test_split(
    X, y_class, test_size=0.2, random_state=6, stratify=y_class
)

# Model setup and training
clf = RandomForestClassifier(
    n_estimators=250,
    random_state=6,
    class_weight='balanced'
)
clf.fit(x_train, y_class_train)

base_path = ROOT_DIR /'models'

# Evaluation
y_pred = clf.predict(x_test)
report_path = base_path / 'rain_classifier_report.txt'
with open(report_path, 'w') as f:
    f.write(f"Accuracy: {accuracy_score(y_class_test, y_pred)}\n\n")
    f.write(classification_report(y_class_test, y_pred))
    f.write(f'\n\n Confusion Matrix: {confusion_matrix(y_class_test, y_pred)}')

# Save model
joblib.dump(clf, base_path / 'rain_classifier.pkl')

# Feature importance (optional)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importances.to_csv(base_path/'feature_importances.csv', index=False)
