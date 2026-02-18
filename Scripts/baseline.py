import sys
import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
sys.path.append(r'C:\project_5')
from Src.baseline import auto_pipeline


url = r'C:\project_5\Data\Raw_data\Raw_data.csv'
df = pd.read_csv(url)

target = 'Max.Intensity'

# Models with hyperparameters
models = {
    'Random_Forest': RandomForestClassifier(),
    'Desicion_Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(),
    'Logistic_OvR': LogisticRegression(max_iter=1500, random_state=42),
    'Logistic_OvO': LogisticRegression(max_iter=1500, random_state=42)
}

for name, model in models.items():

    if name == 'Logistic_OvR':
        model = OneVsRestClassifier(model)

    if name == 'Logistic_OvO':
        model = OneVsOneClassifier(model)

    baseline = auto_pipeline(df, target=target, model=model, model_name=name)

    baseline.feature_prepare()

    baseline.fit()

    baseline.model_saving()

    baseline.prediction()

    baseline.evaluvation()

    baseline.metrics_saving()