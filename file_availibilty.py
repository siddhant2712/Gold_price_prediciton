import pandas as pd
import os
file_path = r'C:\Users\tanu1\Desktop\Project learning\gold_price_prediction\file.csv'
if os.path.isfile(file_path): 
    data = pd.read_csv(file_path) 
    print('File loaded successfully')
else: 
    print('File not found. Please check the file path.')