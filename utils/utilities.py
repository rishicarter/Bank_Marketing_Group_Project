
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
# bank-additional.csv
def datacleaning_rishi(df):
    '''
    Clean and transform categorical and numerical columns.
    '''
    df=df.copy()

    # Replace 'unkown' with NaN
    df.replace('unknown', np.nan, inplace=True)

    # Clean categorical columns - Impute Missing Values
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode().values[0], inplace=True)

    # Column Transformations
    for col in df.select_dtypes(include=['int64']).columns:
        df[col]=df[col].astype(np.float64)

    for col in ['default', 'housing', 'loan', 'y']:
        df[col]=np.where(df[col]=='yes', 1.0, 0.0)

    for col in df.select_dtypes(include=['object']).columns:
        df[col]=df[col].astype('category')

    # Clean numerical columns - Normalize using MinMax
    for col in df.select_dtypes(include=['float64']).columns:
        min_value = df[col].min()
        max_value = df[col].max()
        df[col]=((df[col] - min_value) / (max_value - min_value))

    return df



# bank-additional-full.csv
def datacleaning_terry(df):
    # replace "unknown" with NA
    df_new = df
    df_new = df_new.replace("unknown", np.nan)
    
    # define target variables
    missing_columns = df_new.columns[df_new.isnull().any()]
    
    # define independent columns and target columns
    df_not_missing = df_new[~df_new[missing_columns].isna().any(axis = 1)]
    df_missing = df_new[df_new[missing_columns].isnull().any(axis=1)]

    # define x variable for training data
    x = df_not_missing.drop(columns = missing_columns)
    x = x.drop(columns = ['y','poutcome'])
    x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns) 

    # define x variable for testing data
    x_missing = df_missing.drop(columns = missing_columns)
    x_missing = x_missing.drop(columns = ['y','poutcome'])
    x_missing = pd.get_dummies(x_missing, columns=x_missing.select_dtypes(include=['object']).columns) 
    
    # Predict and fill the value for NA, one column at a time, for all target columns
    for column in missing_columns:
        print(column)
        df_toFill = df_new[df_new[column].isna()]

        # define y variable for training data, and the column to be filled
        y = df_not_missing[column]
        y_missing = df_missing[column]
        
        model = RandomForestClassifier()
        model.fit(x, y)
        
        missing_values_predicted = model.predict(x_missing)             
        
        columns_to_fill = column
        y_missing = np.where(y_missing.isna(), missing_values_predicted, y_missing)
        
        df_missing[column] = y_missing
        
    #concat the data back together, drop the "poutcome" column, and remove duplicate rows
    df = pd.concat([df_not_missing, df_missing])
    df = df.sort_index()
    df = df.drop(columns = "poutcome")
    df = df.drop_duplicates()
    
    return df
        
        


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
            if type(item[i]) == float:
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




