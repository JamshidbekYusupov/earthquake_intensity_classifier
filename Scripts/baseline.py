import sys
import os 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv(r'C:\project_5\Data\Raw_data\Raw_data.csv')

# df = pd.read_csv(data_path)

target = 'Max.Intensity'

# Models with hyperparameters
models = {
    'Random_Forest': RandomForestClassifier(),
    'Desicion_Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(),
    'Logistic_Reg': LogisticRegression()
}

sys.path.append(r'C:\project_5')

from Src.baseline import auto_pipeline

for name, model in models.items():

    au = auto_pipeline(df, target=target, model=model, model_name=name)

    au.feature_prepare()

    au.fit()

    au.model_saving()

    au.prediction()

    au.evaluvation()

    au.metrics_saving()