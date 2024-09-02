import pandas as pd
import numpy as np

from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm


class explanation_collection:
    def __init__(self):
        self.X = None
        self.X_q = None
        self.model = None

    def predict_proba_with_names(self, x):
        # Convert the NumPy array to DataFrame with the same columns as the training data
        x_df = pd.DataFrame(x, columns=self.X.columns)
        return self.model.predict_proba(x_df)

    def get_explanations(self, training, querying, model):
        self.X = training
        self.X_q = querying
        self.model = model
        
        # Initialize LIME explainers
        explainer = LimeTabularExplainer(
            training_data = self.X.to_numpy(), 
            feature_names = self.X.columns
        )

        # Generate explanations for all test instances
        explanations = []
        
        current_count = 0
        n_test=5000
        num_test_cases = n_test
        
        for i in tqdm(range(self.X_q.shape[0])):
            if current_count >= num_test_cases:
                break
            
            exp = explainer.explain_instance(
                data_row = self.X_q.iloc[i].to_numpy(), 
                predict_fn = self.predict_proba_with_names,
                top_labels = 1,
                num_features = self.X.shape[1]
            )
            
            weights_array = np.zeros(self.X.shape[1])
            top_label = exp.top_labels[0]
        
            exp_map = exp.local_exp[top_label]
            for feat_idx, weight in exp_map:
                weights_array[feat_idx] = weight
        
            intercept = exp.intercept[top_label]
        
            complete_exp = np.append(weights_array, [intercept, top_label])
            explanations.append(complete_exp)
            
            current_count += 1

        return explanations