from pathlib import Path
import pandas as pd
import  os

data_folder = Path(r"C:\Users\aalhossary\OneDrive - wesleyan.edu\HtTYM\TPDB" if os.name == 'nt' else '/smithlab/home/aalhossary/HtTYM/TPDB')
datafile1 = "main.xlsx"
datafile2 = "main-D_GE5.xlsx"

df = pd.read_excel(data_folder/datafile1)
print(df.shape, len(df), df.columns)
files_with_DA_AA = [100001, 120439, 155122, 155871, 157435, 157543, 158532]

df.drop(df[df['Sequence'].str.len() < 5].index, inplace=True)

df = df[~df['ID'].isin(files_with_DA_AA)]
print(df.shape, len(df))

df.drop(df[df['HELM notation'].str.contains('\\[d[A-Z]]', regex=True)].index, inplace=True)
print(df.shape, len(df))

df.drop(df[df['Sequence'].str.contains('[a-z]', regex=True)].index, inplace=True)
print(df.shape, len(df))

df.to_excel(data_folder/datafile2, index=False)
