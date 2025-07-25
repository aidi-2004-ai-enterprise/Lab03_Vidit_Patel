# train.py file
#importing required libraries here
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb
import json
import os

#Loading the dataset
df = sns.load_dataset('penguins')
df.dropna(inplace=True)

#One-hot encode categorical features
categorical_features = ['sex', 'island']
df_encoded = pd.get_dummies(df, columns=categorical_features)

#label encode target variable
le = LabelEncoder()
df_encoded['species'] = le.fit_transform(df_encoded['species'])

#splitting features and target
X = df_encoded.drop('species', axis=1)
y = df_encoded['species']

#splitting data into 80/20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# training XGBoost model
model = xgb.XGBClassifier(
    max_depth=3,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
model.fit(X_train, y_train)

# evaluating the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Train F1-score:", f1_score(y_train, y_train_pred, average='weighted'))
print("Test F1-score:", f1_score(y_test, y_test_pred, average='weighted'))
print("\nClassification Report on Test Data:\n", classification_report(y_test, y_test_pred))

#saving model
model_path = 'app/data'
os.makedirs(model_path, exist_ok=True)
model.save_model(f'{model_path}/model.json')

with open(f'{model_path}/label_encoder_classes.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)
