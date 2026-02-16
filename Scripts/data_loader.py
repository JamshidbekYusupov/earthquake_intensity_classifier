import pandas as pd
import numpy as np 
import sys
import os

data_pathes = r'C:\project_5\Data\Scraped_Data'

sys.path.append(r'C:\project_5')

from Src.data_loader import getting_data

gd = getting_data(data_pathes)

gd.data_concating().date_handle().long_latitude().data_saving()