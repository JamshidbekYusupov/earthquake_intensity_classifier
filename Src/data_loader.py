import pandas as pd 
import numpy as np
import os
import glob
import logging


log_path = r'C:\project_5\Logging\data_loader.log'

logging.basicConfig(
    filename=log_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s-%(levelname)s-%(message)s'
)

logging.info("Data Loading Boshlandi")


class getting_data:
    def __init__(self, path):
        self.path = path
        self.df = None

    def data_concating(self):

        try:
            files = glob.glob(os.path.join(self.path, "*.csv"))
            self.df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
            logging.info(f"Datasets are concatted SUCCEDDFULLY")
            return self
        except Exception as e:
            logging.error(f'Error {e} while concating')
    
    def date_handle(self):
        try:
            self.df['Date'] = pd.to_datetime(
                self.df["Date&Time(KST)"],
                format="%Y/%m/%d %H:%M:%S"
            )
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Month'] = self.df['Date'].dt.month
            self.df['Day'] = self.df['Date'].dt.day
            self.df['hour'] = self.df['Date'].dt.hour
            self.df['minute'] = self.df['Date'].dt.minute
            self.df['sec'] = self.df['Date'].dt.second
            self.df = self.df.drop(columns=['Date','Date&Time(KST)', 'Map', 'No.'])
            logging.info(f'Date Handling is DONE')
            return self
        except Exception as e:
            logging.error(f'Error of {e} while date hanling')

    def long_latitude(self):
        try:
            self.df["Longitude"] = self.df["Longitude"].astype(str).str[:6]
            self.df['Longitude'] = self.df["Longitude"].astype(float)

            self.df["Latitude"] = self.df["Latitude"].astype(str).str[:6]
            self.df['Latitude'] = self.df["Latitude"].astype(float)

            self.df['Depth(km)'] = self.df['Depth(km)'].replace('-', 16)
            self.df['Depth(km)'] = self.df["Depth(km)"].astype(float)

            logging.info("Longitute and Latitudes are handlied")
            return self
        
        except Exception as e:
            logging.error(f'Error while handling Longitute and Latitudes {e}')

    def data_saving(self):
        try:
            out_dir = r'C:\project_5\Data\Raw_data'

            os.makedirs(out_dir, exist_ok=True)

            data_path = os.path.join(out_dir, 'Raw_data.csv')
            self.df.to_csv(data_path, index=False)
            logging.info(f'Datset is saved at {out_dir}')
            return self
        except Exception as e:
            logging.error(f'Error while saving dataset {e}')