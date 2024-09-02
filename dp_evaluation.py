import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from differential_privacy import DifferentialPrivacy

class MembershipInferenceAttack:
    def __init__(self, target_model, num_shadow_models=2):
        self.target_model = target_model
        self.num_shadow_models = num_shadow_models
    
    def create_shadow_datasets(self, data, target):
        shadow_datasets = []
        for _ in range(self.num_shadow_models):
            shadow_data_train, shadow_data_test, shadow_target_train, shadow_target_test = train_test_split(
                data, target, test_size=0.5, random_state=42
            )
            shadow_datasets.append((shadow_data_train, shadow_data_test, shadow_target_train, shadow_target_test))
        return shadow_datasets
    
    def train_shadow_models(self, shadow_datasets):
        shadow_models = []
        for shadow_data_train, _, shadow_target_train, _ in shadow_datasets:
            shadow_model = RandomForestClassifier(random_state=42)
            shadow_model.fit(shadow_data_train, shadow_target_train)
            shadow_models.append(shadow_model)
        return shadow_models
    
    def generate_attack_data(self, shadow_models, shadow_datasets, lime_explainer, sample_size):
        attack_data = []
        attack_labels = []
        for i, (shadow_data_train, shadow_data_test, shadow_target_train, shadow_target_test) in enumerate(shadow_datasets):
            shadow_model = shadow_models[i]
            # Sample indices for both probabilities and LIME explanations
            sample_indices_train = np.random.choice(range(len(shadow_data_train)), size=sample_size, replace=False)
            sample_indices_test = np.random.choice(range(len(shadow_data_test)), size=sample_size, replace=False)
            
            # Sampled data
            shadow_data_train_sampled = shadow_data_train.iloc[sample_indices_train]
            shadow_data_test_sampled = shadow_data_test.iloc[sample_indices_test]
    
            # Predicted probabilities for sampled data
            shadow_train_probs = shadow_model.predict_proba(shadow_data_train_sampled)
            shadow_test_probs = shadow_model.predict_proba(shadow_data_test_sampled)
            
            if lime_explainer:
                explainer_train = lime_explainer.get_explanations(shadow_data_train, 
                                                                  shadow_data_train_sampled, 
                                                                  shadow_model)
                explainer_test = lime_explainer.get_explanations(shadow_data_train, 
                                                                 shadow_data_test_sampled, 
                                                                 shadow_model)
                
                # Convert lists of explanations to NumPy arrays
                explainer_train = np.array(explainer_train)
                explainer_test = np.array(explainer_test)
                
                # Combine probabilities with explanations
                combined_train_data = np.hstack([shadow_train_probs, explainer_train])
                combined_test_data = np.hstack([shadow_test_probs, explainer_test])
            else:
                combined_train_data = shadow_train_probs
                combined_test_data = shadow_test_probs

            attack_data.append(np.concatenate([combined_train_data, combined_test_data]))
            attack_labels.append(np.concatenate([np.ones(len(shadow_train_probs)), np.zeros(len(shadow_test_probs))]))

        attack_data = np.vstack(attack_data)
        attack_labels = np.concatenate(attack_labels)
        return attack_data, attack_labels
    
    def train_attack_model(self, attack_data, attack_labels):
        # print(f"Training attack model with {attack_data.shape[1]} features.")

        attack_model = LogisticRegression(max_iter=300, random_state=42)
        attack_model.fit(attack_data, attack_labels)
        return attack_model
    
    def evaluate_attack(self, attack_model, target_data, target_labels, lime_explainer, sample_size):
        target_probs = self.target_model.predict_proba(target_data)
        
        if lime_explainer:
            # Sample indices for LIME explanations
            sample_indices = np.random.choice(range(len(target_data)), size=sample_size, replace=False)
            
            # Generate LIME explanations for the sampled data
            explanations = lime_explainer.get_explanations(target_data.iloc[sample_indices], 
                                                           target_data.iloc[sample_indices], 
                                                           self.target_model)
            
            # Convert explanations to a numpy array
            explanations = np.array(explanations)
            
            # Initialize an array to hold the full combined data
            combined_data = np.zeros((target_probs.shape[0], target_probs.shape[1] + explanations.shape[1]))
            
            # Fill in the probabilities and explanations
            combined_data[:, :target_probs.shape[1]] = target_probs
            combined_data[sample_indices, target_probs.shape[1]:] = explanations
            
            # Evaluate the attack model using the combined data
            attack_predictions = attack_model.predict(combined_data)
        else:
            # If no LIME explanations, use only the predicted probabilities
            attack_predictions = attack_model.predict(target_probs)
        
        # Calculate and return the accuracy of the attack model
        accuracy = accuracy_score(target_labels, attack_predictions)
        return accuracy


class EuclideanDistanceAttack:
    def __init__(self, epsilon, n, num_repeats, numerical_columns, categorical_columns):
        self.epsilon = epsilon
        self.n = n
        self.num_repeats = num_repeats
        self.dp = DifferentialPrivacy(epsilon)
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

    def split_groups(self, X, y):
        X_group_A, X_group_B, y_group_A, y_group_B = train_test_split(X, y, test_size=0.5, random_state=42)
        return X_group_A, X_group_B, y_group_A, y_group_B

    def sample_case_control(self, X_group_A, X_group_B):
        case_group = X_group_A.sample(self.n, random_state=42)
        control_group = X_group_B.sample(self.n, random_state=42)
        return case_group, control_group

    def create_noisy_group_A(self, X_group_A):
        noisy_group_A = self.dp.apply_differential_privacy(X_group_A, self.numerical_columns, self.categorical_columns, round_to_int=True)
        return noisy_group_A

    def calculate_min_distances(self, group, noisy_group_A):
        # distances = cdist(group, noisy_group_A, metric='euclidean')
        distances = cdist(group, noisy_group_A, metric='hamming')
        min_distances = distances.min(axis=1)
        return min_distances

    def determine_threshold(self, min_distances_control):
        threshold = np.quantile(min_distances_control, 0.05)
        return threshold

    def count_below_threshold(self, min_distances_case, threshold):
        count_below_threshold = np.sum(min_distances_case <= threshold)
        ratio_below_threshold = count_below_threshold / self.n
        return ratio_below_threshold

    def perform_attack(self, X, y):
        X_group_A, X_group_B, y_group_A, y_group_B = self.split_groups(X, y)
        case_group, control_group = self.sample_case_control(X_group_A, X_group_B)
        D_eps = self.create_noisy_group_A(X_group_A)
        min_distances_control = self.calculate_min_distances(control_group, D_eps)
        threshold = self.determine_threshold(min_distances_control)
        min_distances_case = self.calculate_min_distances(case_group, D_eps)
        ratio_below_threshold = self.count_below_threshold(min_distances_case, threshold)
        return ratio_below_threshold

    def repeat_attack(self, X, y):
        ratios = []
        for _ in range(self.num_repeats):
            ratio = self.perform_attack(X, y)
            ratios.append(ratio)
        average_ratio = np.mean(ratios)
        return average_ratio