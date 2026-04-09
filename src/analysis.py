import numpy as np
from scipy.stats import linregress

def regression(x, y):
    x = np.log(x)
    y = np.log(y)

    slope, intercept, r, p, stderr = linregress(x, y)

    return {
        "alpha": -slope,
        "r2": r**2,
        "p": p,
        "stderr": stderr
    }

def model_test(df):
    df = df.dropna()

    results = {}

    # Model 1
    results["lambda_only"] = regression(
        df["lambda2"], df["mean_fixation"]
    )

    # Model 2
    results["diameter_over_lambda"] = regression(
        df["diameter"] / df["lambda2"],
        df["mean_fixation"]
    )

    # Model 3
    results["degree_var_over_lambda"] = regression(
        df["degree_var"] / df["lambda2"],
        df["mean_fixation"]
    )

    return results