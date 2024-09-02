import numpy as np
import pandas as pd

class DifferentialPrivacy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def calculate_sensitivity(self, data, numerical_columns):
        sensitivities = {}
        for col in numerical_columns:
            sensitivities[col] = data[col].max() - data[col].min()
        return sensitivities

    def add_laplacian_noise(self, data, numerical_columns, sensitivities, round_to_int=True):
        noisy_data = data.copy()
        for col in numerical_columns:
            sensitivity = sensitivities[col]
            scale = sensitivity / self.epsilon
            rng = np.random.default_rng()
            noise = rng.laplace(loc=0.0, scale=scale, size=noisy_data[col].shape)
            noisy_data[col] += noise
            
            if round_to_int:
                noisy_data[col] = noisy_data[col].round().astype(int)
                min_val, max_val = data[col].min(), data[col].max()
                noisy_data[col] = noisy_data[col].clip(lower=min_val, upper=max_val)
                
        return noisy_data

    def apply_randomized_response(self, data, categorical_columns):
        noisy_data = data.copy()
        noisy_data = noisy_data.reset_index(drop=True)
        for col in categorical_columns:
            unique_values = noisy_data[col].unique()
            k = len(unique_values)
            prob_keep = np.exp(self.epsilon) / (np.exp(self.epsilon) + k - 1)
            # prob_change = 1 / (np.exp(self.epsilon) + k - 1)

            for i in range(len(noisy_data)):
                rng = np.random.default_rng()
                if rng.random() > prob_keep and k > 1:
                    noisy_data.at[i, col] = rng.choice([val for val in unique_values if val != noisy_data.at[i, col]])
        return noisy_data

    def apply_differential_privacy(self, data, numerical_columns, categorical_columns, round_to_int=True):
        sensitivities = self.calculate_sensitivity(data, numerical_columns)

        # Apply Laplacian noise to numerical columns
        noisy_numerical_data = self.add_laplacian_noise(data, numerical_columns, sensitivities, round_to_int=round_to_int)

        # Apply randomized response to categorical columns
        # noisy_categorical_data = self.apply_randomized_response(data, categorical_columns)
        sensitivities = self.calculate_sensitivity(data, categorical_columns)
        noisy_categorical_data = self.add_laplacian_noise(data, categorical_columns, sensitivities, round_to_int=round_to_int)
        
        # Update the original DataFrame with the noisy numerical and categorical data
        noisy_data = data.copy()
        noisy_data[numerical_columns] = noisy_numerical_data[numerical_columns]
        noisy_data[categorical_columns] = noisy_categorical_data[categorical_columns]

        return noisy_data