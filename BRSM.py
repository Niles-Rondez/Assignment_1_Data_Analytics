import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

def balanced_risk_set_matching(df, treatment_col, covariate_cols, caliper=0.1, ratio=1):
    """
    Enhanced Balanced Risk Set Matching with Mahalanobis distance and proper caliper handling
    
    Parameters:
    df (pd.DataFrame): Dataset containing treatment and covariates
    treatment_col (str): Name of treatment column (binary 1/0)
    covariate_cols (list): Covariate columns to balance
    caliper (float): Standardized maximum distance (SD units)
    ratio (int): Control:treated ratio (default 1:1)
    
    Returns:
    pd.DataFrame: Matched dataset with original indices
    pd.DataFrame: Balance diagnostics report
    """
    
    # Preprocessing
    df = df.dropna(subset=[treatment_col] + covariate_cols).copy()
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Insufficient treatment/control units")
    
    # Standardize covariates using control group parameters
    scaler = StandardScaler().fit(control[covariate_cols])
    df_std = df.copy()
    df_std[covariate_cols] = scaler.transform(df[covariate_cols])
    
    # Estimate propensity scores using logistic regression
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(df_std[covariate_cols], df[treatment_col])
    df_std['ps'] = ps_model.predict_proba(df_std[covariate_cols])[:, 1]
    
    # Calculate Mahalanobis distance matrix
    cov = EmpiricalCovariance().fit(control[covariate_cols]).covariance_
    inv_cov = np.linalg.pinv(cov)
    
    # Create neighbor search structure
    nbrs = NearestNeighbors(
        n_neighbors=ratio,
        metric='mahalanobis',
        metric_params={'VI': inv_cov}
    ).fit(control[covariate_cols])
    
    # Find matches with caliper constraint
    distances, indices = nbrs.kneighbors(treated[covariate_cols])
    
    # Apply caliper in standardized units
    valid_matches = (distances <= caliper).any(axis=1)
    treated_matched = treated[valid_matches]
    control_matched = control.iloc[indices[valid_matches].flatten()]
    
    # Create matched dataset
    matched_df = pd.concat([
        treated_matched,
        control_matched.assign(matched_pair=np.repeat(treated_matched.index, ratio))
    ]).sort_index()
    
    # Balance diagnostics
    balance_report = pd.DataFrame({
        'Variable': covariate_cols + ['ps'],
        'StdDiff Before': calculate_std_diff(df, treatment_col, covariate_cols + ['ps']),
        'StdDiff After': calculate_std_diff(matched_df, treatment_col, covariate_cols + ['ps'])
    })
    
    return matched_df, balance_report

def calculate_std_diff(df, treatment_col, variables):
    means = df.groupby(treatment_col)[variables].mean()
    stds = df.groupby(treatment_col)[variables].std()
    return (means.loc[1] - means.loc[0]) / np.sqrt((stds.loc[1]**2 + stds.loc[0]**2)/2)

# Example Usage
# df = pd.read_csv("your_data.csv")
# matched_data, diagnostics = balanced_risk_set_matching(
#     df, 
#     treatment_col="treatment",
#     covariate_cols=["age", "income", "education"],
#     caliper=0.2,
#     ratio=1
# )
# print("Matched Data:\n", matched_data)
# print("\nBalance Diagnostics:\n", diagnostics)