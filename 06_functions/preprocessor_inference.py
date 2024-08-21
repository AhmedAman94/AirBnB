import pandas as pd
import numpy as np

def preprocess_data(df, train_df=None):
    """
    Preprocess Strategy: 
    Drop extremely high null features. (missing > 75% of data in train set)
    Define numerical, categorical, and date time features and convert to appropriate dtype.
    Define irrelevant features and drop them from the data.
    Impute missing values. (median value for numerical for skewness resistance and unknown value for categorical features)
    Calculate time deltas for datetime features.
    """

    # Step 0: Separate the 'id' column
    id_column = df['id'] if 'id' in df.columns else None

    # Step 1: Drop extremely high null features (missing > 75% in train set)
    if train_df is not None:
        high_null_features = train_df.columns[train_df.isnull().mean() > 0.75]
        df = df.drop(columns=high_null_features)
    else:
        high_null_features = df.columns[df.isnull().mean() > 0.75]
        df = df.drop(columns=high_null_features)
    
    # Step 2: Define numerical, categorical, and datetime features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    datetime_features = ['last_scraped', 'host_since'] 
    
    # Convert data types
    for col in datetime_features:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Step 3: Define and drop irrelevant features
    irrelevant_features = ['id', 'listing_url', 'scrape_id', 'picture_url', 
                           'host_url', 'host_thumbnail_url', 'host_picture_url']
    df = df.drop(columns=[col for col in irrelevant_features if col in df.columns])
    
    # Step 4: Impute missing values
    # Impute numerical features with the median
    for col in numerical_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Impute categorical features with 'Unknown'
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)
    
    # Step 5: Calculate time deltas for datetime features
    reference_date = pd.to_datetime('today')
    for col in datetime_features:
        if col in df.columns:
            df[f'{col}_delta'] = (reference_date - df[col]).dt.days
    
    # Drop original datetime columns 
    df = df.drop(columns=datetime_features)
    
    # Step 6: Reattach the 'id' column if it exists
    if id_column is not None:
        df.insert(0, 'id', id_column)
    
    return df

def create_dummy_features(df, categorical_features=None):
    """
    Create dummy variables for specified categorical features and drop the original categorical columns.
    """
    # If no specific categorical features are provided, use all object-type columns
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['category', 'object']).columns.tolist()

    # Create dummy variables only for the specified categorical features
    df_with_dummies = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    return df_with_dummies

def align_columns(train_df, test_df):
    """
    Align the columns of the test set with the training set.

    """
    # Align the test set columns with the training set columns
    test_df_aligned = test_df.reindex(columns=train_df.columns, fill_value=0)
    
    return test_df_aligned
