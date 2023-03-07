import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Get Data
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\IE 517\\Group Project\\MLF_GP1_CreditScore.csv')

#Seperate Data
x_values = Data.drop(columns = ["InvGrd","Rating"])
y_values_I = Data['InvGrd']
y_values_R = Data['Rating']

#EDA
##EDA 01: Size:
count_row = Data.shape[0]  # Gives number of rows
count_col = Data.shape[1]  # Gives number of columns
print('The number of rows in the data frameis %i' %count_row )
print('The number of columns in the data frame is %i' %count_col)
##EDA02:Empty values?
print('\n')
print('empty values in columns?')
print(Data[Data.isnull().T.any()]) 

##EDA 02: Nature of Attributes:
columns_cat = Data.applymap(np.isreal).all()
print('\n')
print('Are the columns numerical or not? (True for numerical)')
print(columns_cat)

##EDA 03:Coefficient&heatmap
corr = x_values.corr()
sns.heatmap(corr, cmap='coolwarm')

#Feature selections
LDA_I = LinearDiscriminantAnalysis(n_components=1)
x_lda_I = LDA_I.fit_transform(x_values,y_values_I)

counts_labels = y_values_R.value_counts()
n_labels =counts_labels.shape[0]


LDA_R = LinearDiscriminantAnalysis(n_components=(n_labels-1))
x_lda2_R = LDA_R.fit_transform(x_values,y_values_R)
