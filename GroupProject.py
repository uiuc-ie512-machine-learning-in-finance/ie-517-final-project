import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Get Data
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\Group Project\\MLF_GP1_CreditScore.csv')

#Seperate Data
x_values = Data.drop(columns = ["InvGrd","Rating"])
y_values_I = Data['InvGrd']
y_values_R = Data['Rating']

#EDA
#EDA 01: Size:
count_row = Data.shape[0]  # Gives number of rows
count_col = Data.shape[1]  # Gives number of columns
print('The number of rows in the data frameis %i' %count_row )
print('The number of columns in the data frame is %i' %count_col)
#EDA02:Empty values?
print('\n')
print('empty values in columns?')
print(Data[Data.isnull().T.any()]) 

#EDA 02: Nature of Attributes:
columns_cat = Data.applymap(np.isreal).all()
print('\n')
print('Are the columns numerical or not? (True for numerical)')
print(columns_cat)

#Coefficient&heatmap
corr = x_values.corr()
sns.heatmap(corr, cmap='coolwarm')
#Feature selections