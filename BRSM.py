import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def balanced_risk_set_matching(df, treatment_col, covariate_cols, caliper=0.1):
    """
    Perform Balanced Risk Set Matching.
    
    Parameters:
    df (pd.DataFrame): The dataset containing treatment and covariate columns.
    treatment_col (str): The name of the treatment column (1 for treated, 0 for control).
    covariate_cols (list): List of covariate column names to balance.
    caliper (float): Maximum allowed difference in propensity scores.
    
    Returns:
    pd.DataFrame: Matched dataset.
    """
    
    # Step 1: Standardize Covariates
    scaler = StandardScaler()
    df[covariate_cols] = scaler.fit_transform(df[covariate_cols])
    
    # Step 2: Estimate Propensity Scores
    model = LogisticRegression()
    model.fit(df[covariate_cols], df[treatment_col])
    df["propensity_score"] = model.predict_proba(df[covariate_cols])[:, 1]
    
    # Step 3: Separate Treated and Control Groups
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    # Step 4: Match Using Nearest Neighbor
    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(control[["propensity_score"]])
    distances, indices = nbrs.kneighbors(treated[["propensity_score"]])
    
    matched_control_indices = indices.flatten()
    matched_control = control.iloc[matched_control_indices].reset_index(drop=True)
    
    # Apply Caliper Constraint
    matched_control = matched_control[np.abs(matched_control["propensity_score"] - treated["propensity_score"].values) <= caliper]

    # Step 5: Return Matched Dataset
    matched_df = pd.concat([treated.reset_index(drop=True), matched_control.reset_index(drop=True)], axis=0)
    return matched_df

# Example Usage
# df = pd.read_csv("your_dataset.csv")
# matched_data = balanced_risk_set_matching(df, treatment_col="treatment", covariate_cols=["age", "income", "education"])
# print(matched_data)