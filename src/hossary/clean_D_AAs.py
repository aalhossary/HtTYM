import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\aalhossary\OneDrive - wesleyan.edu\HtTYM\TPDB\main.xlsx")
print(df.shape, len(df), df.columns)
files_with_DA_AA = [100001, 120439, 155122, 155871, 157435, 157543, 158532]

df.drop(df[df['Sequence'].str.len() < 5].index, inplace=True)

df = df[~df['ID'].isin(files_with_DA_AA)]
print(df.shape, len(df))

df.drop(df[df['HELM notation'].str.contains('\\[d[A-Z]]', regex=True)].index, inplace=True)
print(df.shape, len(df))

df.drop(df[df['Sequence'].str.contains('[a-z]', regex=True)].index, inplace=True)
print(df.shape, len(df))

df.to_excel(r"C:\Users\aalhossary\OneDrive - wesleyan.edu\HtTYM\TPDB\main-D_GE5.xlsx", index=False)
