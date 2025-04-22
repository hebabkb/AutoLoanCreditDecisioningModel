
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(train_path, eval_path):
    df_train = pd.read_csv(train_path)
    df_eval = pd.read_csv(eval_path)
    return df_train, df_eval

def cap_outliers(df, numerical_features, lower_quantile=0.01, upper_quantile=0.99):
    for col in numerical_features:
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        df[col] = np.clip(df[col], lower, upper)
    return df

def preprocess_data(df_train, df_eval, correlation_threshold=0.7, missing_threshold=0.8):
    # Flip bad_flag
    df_train['bad_flag'] = df_train['bad_flag'].map({1: 0, 0: 1})
    df_eval['bad_flag'] = df_eval['bad_flag'].map({1: 0, 0: 1})

    # Drop rows with missing targets
    df_train.dropna(subset=['bad_flag'], inplace=True)
    df_eval.dropna(subset=['bad_flag'], inplace=True)

    # Drop features with too many missing values
    cols_to_drop = df_train.columns[df_train.isnull().mean() > missing_threshold].tolist()
    cols_to_drop += ['aprv_flag', 'Gender', 'Race']
    df_train.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df_eval.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Feature types
    categorical_features = ['collateral_dlrinput_newused_1req']
    numerical_features = df_train.select_dtypes(include=['float64', 'int64']).columns.drop('bad_flag', errors='ignore')

    # Remove correlated features
    corr_matrix = df_train[numerical_features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper_triangle.columns if any(upper_triangle[col] > correlation_threshold)]
    numerical_features = numerical_features.drop(to_drop_corr)

    # Cap outliers before scaling
    df_train = cap_outliers(df_train, numerical_features)
    df_eval = cap_outliers(df_eval, numerical_features)

    # Pipelines
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Training data
    X_train = df_train[numerical_features.tolist() + categorical_features]
    y_train = df_train['bad_flag'].astype(int)
    X_train_processed = preprocessor.fit_transform(X_train)

    # Evaluation data
    for col in numerical_features.tolist() + categorical_features:
        if col not in df_eval.columns:
            df_eval[col] = 'Unknown' if col in categorical_features else np.nan

    X_eval = df_eval[numerical_features.tolist() + categorical_features]
    y_eval = df_eval['bad_flag'].astype(int)
    X_eval_processed = preprocessor.transform(X_eval)

    return X_train_processed, y_train, X_eval_processed, y_eval, to_drop_corr, cols_to_drop, preprocessor

if __name__ == "__main__":
    train_path = "/Users/heba/Desktop/Erdos/Training Dataset A_R-384891_Candidate Attach #1_PresSE_SRF #1142.csv"
    eval_path = "/Users/heba/Desktop/Erdos/Evaluation Dataset B_R-384891_Candidate Attach #2_PresSE_SRF #1142.csv"

    df_train, df_eval = load_data(train_path, eval_path)
    X_train_processed, y_train, X_eval_processed, y_eval, to_drop_corr, cols_to_drop = preprocess_data(df_train, df_eval)

    print("Train shape:", X_train_processed.shape)
    print("Eval shape:", X_eval_processed.shape)
    print("Dropped correlated features:", to_drop_corr)
    print("Dropped for missingness/leakage/sensitivity:", cols_to_drop)
