import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
from imblearn.over_sampling import SMOTE

log_path = r'C:\project_5\Logging\baseline.log'


logging.basicConfig(
    filename= log_path,
    filemode='a',
    level= logging.INFO,
    format = '%(asctime)s-%(levelname)s-%(message)s'
)

logging.info('Pipeline Building has been started')

class auto_pipeline(BaseEstimator):

    def __init__(self, df:pd.DataFrame, target:str, model, model_name):
        self.df = df.copy()
        self.target = target
        self.model_algorithm = model
        self.model_name = model_name
        self.models = {}
        self.metrics = {}
        self.model = model
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def feature_prepare(self):

        try:
            #  Dropping NaN values, as this is baseline
            self.df = self.df.dropna(subset=[self.target])
            X = self.df.drop(columns=self.target)
            y = self.df[self.target]

            maps = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ':2, 'Ⅳ': 3, 'Ⅴ': 4}
            y = y.map(maps)
            num_cols = X.select_dtypes(include=[np.number]).columns.to_list()
            cat_cols = X.select_dtypes(exclude=[np.number]).columns.to_list()

            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='mean'))
            ])

            logging.info('Numerical pipeline is DONE')

            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            logging.info('Categorical pipeline is DONE')
            self.preprocessor = ColumnTransformer([
                ('num', num_pipe, num_cols),
                ('cat', cat_pipe, cat_cols)
            ])
            logging.info('Column Transforming is done')
            logging.info('PIPELINE building is DONE SUCCESSFULLY')
            return X, y
        
        except Exception as e:
            logging.error(f'ERROR while building PIPELINE: {e}')
            raise
    
    def fit(self):
        try:

            X, y = self.feature_prepare()

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,test_size=0.2, random_state=42)

            if self.model_algorithm is None:
                raise ValueError('Model algorithm is not specified, please specify model first')
            
            self.model = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', self.model_algorithm)
            ])
            self.model.fit(self.X_train, self.y_train)

            logging.info('Fittig model is done')

            return self
        except Exception as e:
            logging.error(f'ERROR while fitting the model:{e}')
            raise
    
    def model_saving(self):
        out_dir = r'C:\project_5\Model\All_Model'
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'{self.model_name}.joblib')
        joblib.dump(self.model, path)

        return self
 
    def prediction(self):
        try:
            self.y_pred = self.model.predict(self.X_test)
            logging.info(f'Prediction is done with model {self.model_name}')
            return self
        except Exception as e:
            logging.error(f'ERROR while predicting for {self.model_name}, error:{e}')
            raise
    
    def evaluvation(self):
        try:
            self.metrics[f'{self.model_name}_accuracy'] = accuracy_score(self.y_test, self.y_pred)
            self.metrics[f'{self.model_name}_precison'] = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=True)
            self.metrics[f'{self.model_name}_recall'] = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=True)
            self.metrics[f'{self.model_name}_f1'] = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=True)
        
            metrics = [[f'{self.metrics[f'{self.model_name}_accuracy']}', f'{self.metrics[f'{self.model_name}_precison']}', f'{self.metrics[f'{self.model_name}_recall']}'], f'{self.metrics[f'{self.model_name}_f1']}']

            headers = ['Accuracy Score', 'Precision Score', 'Recall Score', 'F1 Score']

            print(tabulate(metrics, headers, floatfmt='.4f', tablefmt='orgtbl'))

            logging.info(f'Metrics are calculated SUCCESSFFULLY for {self.model_name}')
            return self
        except Exception as e:
            logging.error(f'ERROR while calculating metrics for {self.model_name}')
            raise
    
    def metrics_saving(self):
        try:
            # to DataFrame
            df = pd.DataFrame([self.metrics])
            
            #output path
            output_dir = r'C:\project_5\Metrics'

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{self.model_name}_metrics.csv')
            # DataFrame to CSV
            df.to_csv(output_file, index=False)
            logging.info(f'Metrics are savded at {output_dir}')
            return self
        except Exception as e:
            logging.error(f'ERROR while saving metics:{e}')
            raise