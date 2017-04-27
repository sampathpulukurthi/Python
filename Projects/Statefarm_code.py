# Importing modules
import pandas as pd
import numpy as np
import os
import matplotlib as plt
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Read Dataset
Train_data = pd.read_csv("E:/Interview Challenge/Statefarm/Dataset/Data for Cleaning & Modeling.csv",
                         sep=',', index_col=False, dtype='unicode')
Test_data = pd.read_csv("E:/Interview Challenge/Statefarm/Dataset/Holdout for Testing.csv",
sep=',', index_col=False, dtype='unicode')

# Finding the train and test dimensions
print(Train_data.shape)
print(Test_data.shape)

# First few rows of the dataset
Train_data.head(3)                      

# Finding the missing values percentage each variable (Train)
table_cols = ["Feature","#missing_values","Per_missing"]
Summary_Train = pd.DataFrame(index=range(Train_data.shape[1]), columns=[table_cols])

# Train missing values
i=0
for col in Train_data.columns:
    Summary_Train.loc[i,table_cols[0]] = col
    Summary_Train.loc[i,table_cols[1]] = Train_data[col].isnull().sum()
    Summary_Train.loc[i,table_cols[2]] = (float(Train_data[col].isnull().sum())/(Train_data.shape[0]))*100
    i = i+1
    
Summary_Train

# Finding the missing values percentage each variable (Test)
Summary_Test = pd.DataFrame(index=range(Test_data.shape[1]), columns=[table_cols])

# Test missing values
i=0
for col in Test_data.columns:
    Summary_Test.loc[i,table_cols[0]] = col
    Summary_Test.loc[i,table_cols[1]] = Test_data[col].isnull().sum()
    Summary_Test.loc[i,table_cols[2]] = (float(Test_data[col].isnull().sum())/(Test_data.shape[0]))*100
    i = i+1
    
Summary_Test.head(2)

# Preprocess Train Data columns X1,X4,X5,X6,X7,X21,X30
Train_data[['X1', 'X30','X21']] = Train_data[['X1', 'X30','X21']].replace('%','',regex=True).astype('float')
Train_data[['X4', 'X5','X6']] = Train_data[['X4', 'X5','X6']].replace(to_replace=[','], value='', regex=True)
Train_data[['X4', 'X5','X6']]= Train_data[['X4', 'X5','X6']].replace('[\$,)]','', regex=True).astype(float)
Train_data['X7'] = Train_data[['X7']].replace(to_replace='months', value='', regex=True)
Train_data.head(3)


# Preprocess Test Data columns X1,X4,X5,X6,X7,X21,X30
Test_data[['X1', 'X30','X21']] = Test_data[['X1', 'X30','X21']].replace('%','',regex=True).astype('float')
Test_data[['X4', 'X5','X6']] = Test_data[['X4', 'X5','X6']].replace(to_replace=[','], value='', regex=True)
Test_data[['X4', 'X5','X6']]= Test_data[['X4', 'X5','X6']].replace('[\$,)]','', regex=True).astype(float)
Test_data['X7'] = Test_data[['X7']].replace(to_replace='months', value='', regex=True)
Test_data.head(3)

# converting X11 years of experience to numeric (Train Data)
Train_data.loc[Train_data.X11 == '< 1 year', 'X11'] = 0
Train_data.loc[Train_data.X11 == 'n/a', 'X11'] = 0
Train_data.loc[Train_data.X11 == '1 year', 'X11'] = 1
Train_data.loc[Train_data.X11 == '2 years', 'X11'] = 2
Train_data.loc[Train_data.X11 == '3 years', 'X11'] = 3
Train_data.loc[Train_data.X11 == '4 years', 'X11'] = 4
Train_data.loc[Train_data.X11 == '5 years', 'X11'] = 5
Train_data.loc[Train_data.X11 == '6 years', 'X11'] = 6
Train_data.loc[Train_data.X11 == '7 years', 'X11'] = 7
Train_data.loc[Train_data.X11 == '8 years', 'X11'] = 8
Train_data.loc[Train_data.X11 == '9 years', 'X11'] = 9
Train_data.loc[Train_data.X11 == '10+ years', 'X11'] = 10

# converting X11 years of experience to numeric (Test Data)
Test_data.loc[Test_data.X11 == '< 1 year', 'X11'] = 0
Test_data.loc[Test_data.X11 == 'n/a', 'X11'] = 0
Test_data.loc[Test_data.X11 == '1 year', 'X11'] = 1
Test_data.loc[Test_data.X11 == '2 years', 'X11'] = 2
Test_data.loc[Test_data.X11 == '3 years', 'X11'] = 3
Test_data.loc[Test_data.X11 == '4 years', 'X11'] = 4
Test_data.loc[Test_data.X11 == '5 years', 'X11'] = 5
Test_data.loc[Test_data.X11 == '6 years', 'X11'] = 6
Test_data.loc[Test_data.X11 == '7 years', 'X11'] = 7
Test_data.loc[Test_data.X11 == '8 years', 'X11'] = 8
Test_data.loc[Test_data.X11 == '9 years', 'X11'] = 9
Test_data.loc[Test_data.X11 == '10+ years', 'X11'] = 10

# Subset the Train rows where the dependent variable(X1) is missing
Train_data = Train_data[~Train_data['X1'].isnull()]
Train_data.shape

# Convert to numeric variables (Train)
Train_data[['X5','X6', 'X11','X13','X21','X22','X24',
                'X25','X27','X28','X29','X30','X31']] = Train_data[['X5','X6', 'X11','X13','X21','X22','X24',
               'X25','X27','X28','X29','X30','X31']].astype('float')

# Convert to numeric variables (Test)
Test_data[['X5','X6', 'X11','X13','X21','X22','X24',
                'X25','X27','X28','X29','X30','X31']] = Test_data[['X5','X6', 'X11','X13','X21','X22','X24',
                'X25','X27','X28','X29','X30','X31']].astype('float')

				
############# Feature Extraction##################################
dt1=pd.DataFrame(Train_data.X15.str.split('-').apply(pd.Series).astype(str))
dt1.columns=['X15yr','X15mon']

# Date1 Test Data
dt1_test=pd.DataFrame(Test_data.X15.str.split('-').apply(pd.Series).astype(str))
dt1_test.columns=['X15yr','X15mon']

dt2=pd.DataFrame(Train_data.X23.str.split('-').apply(pd.Series).astype(str))
dt2.columns=['X23mon','X23yr']

dt2_test=pd.DataFrame(Test_data.X23.str.split('-').apply(pd.Series).astype(str))
dt2_test.columns=['X23mon','X23yr']

# Create a variable (AmountFunded)/(Income of Borrower)
IncomePer = pd.DataFrame(((Train_data.X5)/(Train_data.X13))*100)
IncomePer.columns=['IncLnratio']
#IncomePer.head(2)

# Create a variable (AmountFunded)/(Income of Borrower) (Test Dataset)
IncomePer_test = pd.DataFrame(((Test_data.X5)/(Test_data.X13))*100)
IncomePer_test.columns=['IncLnratio']
#IncomePer_test.head(2)

# Create a variable InvestorFunding(X6)/Loan Amount Funded(X5)
FundPer = pd.DataFrame(((Train_data.X6)/(Train_data.X5))*100)
FundPer.columns=['FundLnratio']
#FundPer.head(2)

# Create a variable InvestorFunding(X6)/Loan Amount Funded(X5)
FundPer_test = pd.DataFrame(((Test_data.X6)/(Test_data.X5))*100)
FundPer_test.columns=['FundLnratio']
#FundPer_test.head(2)

# Considering variable X25 as it is very significant in determining interest rate
#Train_data.dtypes
X25new = pd.DataFrame(Train_data.X25.fillna(-999))
X25new.columns=['X25new']

# Considering variable X25 as it is very significant in determining interest rate
#Train_data.dtypes
X25new_test = pd.DataFrame(Test_data.X25.fillna(-999))
X25new_test.columns=['X25new']

# Dropping variables not required for analysis
# Train Dataset
Train_data_sub = Train_data.drop(['X2','X3','X10','X15','X16','X18','X19',
                                  'X23','X25','X26'],axis=1) 

print ("Train shape is : ",Train_data_sub.shape)

# Test Dataset
Test_data_sub = Test_data.drop(['X2','X3','X10','X15','X16','X18','X19',
                                'X23','X25','X26'],axis=1) 

print ("Test shape is : ",Test_data_sub.shape)

# Combine all the variable created with Train_data_sub
Train_data_sub = pd.concat([Train_data_sub,dt1,dt2,IncomePer,FundPer,X25new],axis=1)

# Combine all the variable created with Test_data_sub
Test_data_sub = pd.concat([Test_data_sub,dt1_test,dt2_test,IncomePer_test,FundPer_test,X25new_test],axis=1)

# Check for missing values in the data
Train_data_sub.apply(lambda x: sum(x.isnull().values), axis = 0)
# Check for missing values in the data
Test_data_sub.apply(lambda x: sum(x.isnull().values), axis = 0)

## Filling missing values (Train)

# Replace missing values with mode (Categorical)
Train_data_sub.X8[ Train_data_sub.X8.isnull() ] = Train_data_sub.X8.dropna().mode().values
Train_data_sub.X9[ Train_data_sub.X9.isnull() ] = Train_data_sub.X9.dropna().mode().values
Train_data_sub.X12[ Train_data_sub.X12.isnull() ] = Train_data_sub.X12.dropna().mode().values

# Replace missing values with median (Numeric)
Train_data_sub['X13'][ np.isnan(Train_data_sub['X13']) ] = Train_data_sub['X13'].median()
Train_data_sub['X30'][ np.isnan(Train_data_sub['X30']) ] = Train_data_sub['X30'].median()
Train_data_sub['IncLnratio'][ np.isnan(Train_data_sub['IncLnratio']) ] = Train_data_sub['IncLnratio'].median()
Train_data_sub['FundLnratio'][ np.isnan(Train_data_sub['FundLnratio']) ] = Train_data_sub['FundLnratio'].median()

## Filling missing values (Test)
Test_data_sub['X30'][np.isnan(Test_data_sub['X30']) ] = Test_data_sub['X30'].median()


# Check missing value count in the data after imputation
Train_data_sub.apply(lambda x: sum(x.isnull().values), axis = 0)

# Drop NA's if there are any
Train_data_sub.dropna(inplace=True)
Test_data_sub.dropna(inplace=True)

############## Data Exploration########################
import matplotlib.pyplot as plt
%matplotlib inline

def histogram_chart(plt, col, Ylabel="Frequency", Xlabel=None, Title="Histogram"):
    col.dropna(inplace=True)
    
    plt.hist(col)
    
    if Ylabel:
        plt.ylabel(Ylabel)
    
    if Xlabel:
        plt.xlabel(Xlabel)
    
    plt.title(Title)        


# Distribution of Interest Rate
histogram_chart(plt,Train_data_sub.X1,Title = "Interest Rate")

# Scatter plot
#Function to plot a scatter chart
def scatter_chart(plt, col1, col2, Title="Scatter Plot"):
    color = ['r']
    results = linregress(col1,col2)
    print results
    plt.scatter(col1,col2)
    plt.plot(col1, col1*results[0] + results[1])
    plt.ylabel(col2.name)
    plt.xlabel(col1.name)
    plt.title(Title)

	
scatter_chart(plt,Train_data_sub.X1,Train_data_sub.X4)
scatter_chart(plt,Train_data_sub.X4,Train_data_sub.X5)

import seaborn as sns
# Count of each level (bar plot)
ax = sns.countplot(x="X8", data=Train_data_sub)
ax = sns.countplot(x="X11", data=Train_data_sub)

# Label encoding for categorical variables (For Tree based models)

categorical_columns = ['X7','X8','X9',"X12","X14","X17","X20",'X32','X15mon','X23mon','X15yr','X23yr']

#encoding categorical variable
for var in categorical_columns:
    print var
    lb = preprocessing.LabelEncoder()
    lb.fit( Train_data_sub[var] )
    Train_data_sub[var] = lb.transform(Train_data_sub[var].astype('str'))
    #Train_data_sub[var] = lb.transform(test_X[var].astype('str'))

	
# Test Dataset
# Label encoding for categorical variables (For Tree based models) 

categorical_columns = ['X7','X8','X9',"X12","X14","X17","X20",'X32','X15mon','X23mon','X15yr','X23yr']

#encoding categorical variable
for var in categorical_columns:
    print var
    lb = preprocessing.LabelEncoder()
    lb.fit( Test_data_sub[var] )
    Test_data_sub[var] = lb.transform(Test_data_sub[var].astype('str'))

	
# Target variable
Target_var = np.array(Train_data_sub["X1"])

## Dropping the target variable from the data ##
Train_data_sub = Train_data_sub.drop(['X1'],axis=1) 

# Convert to arrays
Final_Train_data = Train_data_sub.values


Test_data_sub = Test_data_sub.drop(['X1'],axis=1) 

Final_Test_data = Test_data_sub.values

#Import Randomforest Library
from sklearn.ensemble import RandomForestRegressor
model= RandomForestRegressor(n_estimators=100,oob_score = True,n_jobs = -1,max_features="auto",min_samples_split=50)

# Train the model using the training sets and check score
model.fit(Final_Train_data,Target_var)

#Predict Output
predicted_rf_test = model.predict(Final_Test_data)
predicted_rf_test.shape

## Gradient boosting machine (GBM algorithm)
from sklearn.ensemble import GradientBoostingRegressor
gbm_model = GradientBoostingRegressor(loss='ls',n_estimators=50,min_samples_split=50,max_depth=5,max_features="sqrt")

# Train the model using the training sets and check score
gbm_model.fit(Final_Train_data,Target_var)

#Predict Output
predicted_gbm_test = gbm_model.predict(Final_Test_data)
predicted_gbm_test.shape

Final_predictions = pd.DataFrame(predicted_gbm_test,predicted_rf_test)

Final_predictions.to_csv("E:/Interview Challenge/Statefarm/Final submission/Results from sampath.csv")









