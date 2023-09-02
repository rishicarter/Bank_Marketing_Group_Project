
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

#bank-full.csv
def binay_col(df, col_name):
    d1 = df.copy() 
    d1[col_name] = (d1[col_name] == 'yes').astype('int64')
    return d1
def datacleaning_yuheng(df):
    d1 = df.copy()

    ## replace 'unknown' values
    d1 = d1.replace("unknown", np.nan)
    d1 = d1.drop(columns='poutcome')
    d1_testing = d1[d1.isnull().any(axis=1)]
    d1_training = d1[d1.notnull().all(axis=1)]

    for col_name in ['default', 'housing', 'loan']:
        d1_training = binay_col(d1_training, col_name)
        d1_testing = binay_col(d1_testing, col_name)

    ## predict null values in columns: job, education, contact
    null_columns = ['job','education', 'contact']
    X = d1_training.drop(columns = null_columns).drop(columns = 'y')
    y = d1_training[null_columns]
    X_test = d1_testing.drop(columns = null_columns).drop(columns = 'y')
    y_test = d1_testing[null_columns]

    X_train_transformed = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns) 
    X_test_transformed = pd.get_dummies(X_test, columns=X_test.select_dtypes(include=['object']).columns) 

    ## use random forest model to predict nan values
    model = RandomForestClassifier()
    model.fit(X_train_transformed, y)

    ## replace only nan values from predictions, not using all predictions 
    for j in range(0, len(y_test.values)):
        item = y_test.values[j]
        for i in range(0, len(item)):
            if type(item[i]) != 'str':
                item[i] = prediction[j][i]
    d1_testing[null_columns] = y_test
    d1 = pd.concat([d1_testing, d1_training])

    
    ## Detect and remove outliers in duration column
    # Calculate the upper and lower limits
    Q1 = d1['duration'].quantile(0.25)
    Q3 = d1['duration'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
 
    # Remove outliers
    d1 = d1[d1['duration'] <= upper].reset_index(inplace= False)
    d1 = d1.drop_duplicates()
    return d1

