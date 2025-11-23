import sys
import os
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

from src.preprocessing.preprocess_train import OUT_TRAIN, OUT_TEST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import joblib


# Load data train dan test
train = pd.read_csv(OUT_TRAIN)
test =  pd.read_csv(OUT_TEST)

#Split
x_train = train.drop(columns='RainTomorrow')
y_train = train['RainTomorrow']

x_test = test.drop(columns='RainTomorrow')
y_test = test['RainTomorrow']

#setup model
clf = RandomForestClassifier(
    n_estimators= 250,
    random_state= 6,
    class_weight='balanced'
)

clf.fit(x_train, y_train)

#prediksi
y_pred = clf.predict(x_test)

#eval
model_path = Path(os.path.join(ROOT, 'models'))
model_path.mkdir(exist_ok=True)

report_path = Path(os.path.join(model_path, 'rain_clf_report.txt'))
with open(report_path, 'w') as f:
    f.write(f'accuracy: {accuracy_score(y_test, y_pred)}\n\n')
    f.write(classification_report(y_test, y_pred))
    f.write(f'\n\nconfusion matrix: \n{confusion_matrix(y_test, y_pred)}')

#save model
model = Path(os.path.join(model_path, 'rain_clf.pkl'))
joblib.dump(clf, model)

#kepentingan features dalam model
features_importances = Path(os.path.join(model_path, 'features_importances.csv'))

features_imp = pd.DataFrame({
    'feature':x_train.columns,
    'importances': clf.feature_importances_
}).sort_values(by='importances', ascending= False)

features_imp.to_csv(features_importances, index= False)