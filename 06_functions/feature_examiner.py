import pandas as pd

def identify_categorical_features(df):
    # Identify categorical features (object or category data types)
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Evaluate cardinality of each categorical feature
    cardinality = {col: df[col].nunique() for col in categorical_features}
    
    # Create a DataFrame 
    categorical_summary = pd.DataFrame({
        'Feature': categorical_features,
        'Unique Values': [df[col].nunique() for col in categorical_features],
        'Example Values': [df[col].unique()[:5] for col in categorical_features]
    }).sort_values(by='Unique Values')
    
    return categorical_summary, cardinality

