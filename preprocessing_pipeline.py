import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from scipy import stats
from imblearn.over_sampling import SMOTE


class PreprocessingPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.label_encoder = LabelEncoder()
        self.ordinal_encoder = OrdinalEncoder()
        self.df = None
        self.target_col = None
        self.X = None
        self.y = None
        self.X_train = None
        self.numerical_columns = None
        self.categorical_columns = None

    
    #------Missing Values------#
    def drop_missing_values(self):
        initial_rows = self.df.shape[0]
        self.df = self.df.dropna()
        removed_rows = initial_rows - self.df.shape[0]
        print(f"Missing Values Dropped: {removed_rows} rows removed")
        percentage_dropped = (removed_rows / initial_rows) * 100
        print(f"Percentage of rows dropped due to missing values: {percentage_dropped:.2f}%")
        
    
    #------Encoding------#
    def encode_categorical_variables(self, train=True):
        # Encode categorical variables in X
        cat_cols = self.categorical_columns
        print(f"number of categorical columns '{len(cat_cols)}")
        
        if train:
            self.X[cat_cols] = self.ordinal_encoder.fit_transform(self.X[cat_cols])
        else:
            self.ordinal_encoder.fit(self.X_train[cat_cols])
            self.X[cat_cols] = self.ordinal_encoder.transform(self.X[cat_cols])
        
        # Encode categorical target variable y
        self.y = self.label_encoder.fit_transform(self.y)

    
    #------Duplicates and Inconsistencies------#
    def drop_all_duplicates(self):
        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        removed_rows = initial_rows - self.df.shape[0]
        print(f"Duplicates Dropped: {removed_rows} rows removed")
        percentage_dropped = (removed_rows / initial_rows) * 100
        print(f"Percentage of rows dropped due to duplicates: {percentage_dropped:.2f}%")

    
    #------Outliers------#
    def handle_outliers(self):
        if not self.numerical_columns:
            raise ValueError("Numerical columns must be provided.")
            
        # Initial number of rows
        initial_rows = self.X.shape[0] 
            
        # Calculate z-scores for numerical columns
        z_scores = np.abs(stats.zscore(self.X[self.numerical_columns]))
        
        # Remove rows with outliers
        outliers = (z_scores > 3).any(axis=1)
        X_outlier_removed = self.X[~outliers]
        y_outlier_removed = self.y[~outliers] if self.y is not None else None
        
        # Number of rows removed
        removed_rows = initial_rows - X_outlier_removed.shape[0]
        print(f"Outliers: {removed_rows} rows removed")
        percentage_dropped = (removed_rows / initial_rows) * 100
        print(f"Percentage of rows dropped due to outliers: {percentage_dropped:.2f}%")
        
        self.X = X_outlier_removed
        self.y = y_outlier_removed

    
    #------Scaling------#
    def scale_numerical_features_only(self, train):
        # Encode categorical variables in X
        num_cols = self.numerical_columns
        print(f"number of numerical columns '{len(num_cols)}")
        
        if train:           
            # Original data before scaling
            original_data = self.X[num_cols].copy()
            
            # Fit and transform the data
            self.X[num_cols] = self.scaler.fit_transform(self.X[num_cols])
            
            # Scaled data
            scaled_data = self.X[num_cols]
            
            # Calculate mean relative change for each feature
            relative_change = (scaled_data.values - original_data.values) / (original_data.values+1)
            mean_relative_change = np.mean(np.abs(relative_change), axis=0)

            # Calculate overall mean relative change across all features
            overall_mean_relative_change = np.mean(mean_relative_change)
            
            # Print the mean relative change for each feature and the overall mean relative change
            print("Mean relative change for each feature:\n", pd.Series(mean_relative_change, index=num_cols))
            print("Overall mean relative change:", overall_mean_relative_change)
            
        else:
            self.scaler.fit(self.X_train[num_cols])
            self.X[num_cols] = self.scaler.transform(self.X[num_cols])
            
    
    #------Resampling------#
    def resample_data(self):
        initial_row_count = len(self.X)
        self.X, self.y = self.smote.fit_resample(self.X, self.y)
        final_row_count = len(self.X)
        percentage_change = ((final_row_count - initial_row_count) / initial_row_count) * 100
        print(f"Percentage change in row count due to resampling: {percentage_change:.2f}%")
        
            
    #------PP Steps Selection------#
    def preprocess(self, df, target_col, numerical_columns, categorical_columns, steps, train=True, X_train=None, balance_feature=None, filter_conditions=None):
        self.df = df.copy()
        
        if not train:
            self.X_train = X_train

        self.target_col = target_col
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[target_col]

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        
        for step in steps:
            if step == 'drop_missing_values':
                self.drop_missing_values()

                self.X = self.df.drop(columns=[target_col])
                self.y = self.df[target_col]

            elif step == 'encode_categorical_variables':
                self.encode_categorical_variables(train=train)
                
                # self.df = pd.concat([self.X, self.y], axis=1).reset_index(drop=True)
                
                columns = self.df.columns
                ndr = np.concatenate([self.X.to_numpy(), self.y.reshape(-1, 1)], axis=1)
                self.df = pd.DataFrame(ndr, columns = columns)
                
            elif step == 'drop_all_duplicates':
                self.drop_all_duplicates()
                
                self.X = self.df.drop(columns=[target_col])
                self.y = self.df[target_col]
                
            elif step == 'handle_outliers':
                self.handle_outliers()
                # columns = self.df.columns
                # ndr = np.concatenate([self.X.to_numpy(), np.array(self.y).reshape(-1, 1)], axis=1)
                # self.df = pd.DataFrame(ndr, columns = columns)
                
                self.df = pd.concat([self.X, self.y], axis=1)
 
            elif step == 'scale_numerical_features_only':
                self.scale_numerical_features_only(train=train)
                
                self.df = pd.concat([self.X, self.y], axis=1)
        
            elif step == 'resample_data':
                self.resample_data()

                self.df = pd.concat([self.X, self.y], axis=1)
                
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")
        
        return self.X, self.y