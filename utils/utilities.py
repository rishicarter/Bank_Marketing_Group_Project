
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier




#----
# Data Cleaning function from each member

'''
def datacleaning_<<YOUR_NAME>>():
    
    'Code Logic'
    
    return 0

'''
# bank-additional-full.csv
def datacleaning_terry(df):
    
    df = df.replace("unknown", np.nan)
    
    missing_columns = df.columns[df.isnull().any()]
    
    df_missing = df[df[missing_columns].isnull().any(axis=1)]
    df_not_missing = df[~df[missing_columns].isnull().any(axis=1)]
    df_not_missing_encoded = pd.get_dummies(df_not_missing, columns=df_not_missing.select_dtypes(include=['object']).columns)
    
    X = df_not_missing.drop(missing_columns, axis=1)
    y = df_not_missing[missing_columns]
    
    X = X.drop(columns = ['poutcome','y'])
    X_encode = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns) 
    
    model = RandomForestClassifier()
    model.fit(X_encode, y)
    
    X_missing = df_missing.drop(missing_columns, axis=1)
    X_missing.drop(columns = ['poutcome','y'], inplace = True)
    X_missing_encode = pd.get_dummies(X_missing, columns=X_missing.select_dtypes(include=['object']).columns) 
    
    missing_values_predicted = model.predict(X_missing_encode)
    df_missing[missing_columns] = missing_values_predicted
    
    df_imputed = pd.concat([df_not_missing, df_missing])
    
    df_imputed = df_imputed.drop_duplicates()
    df_imputed = df_imputed.drop(columns = 'poutcome')
    
    return df_imputed

# bank.csv
def datacleaning_justine(df):

    df = df['age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'].str.split(';', expand=True)

    new_column_names = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
    ]

    df.columns = new_column_names

    return df    

