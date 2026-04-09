import numpy as np
from scipy.stats import linregress

def compute_scaling(lambda_vals, fixation_times):
    x = np.log(lambda_vals)
    y = np.log(fixation_times)

    slope, intercept, r, p, stderr = linregress(x, y)

    return {
        "alpha": -slope,
        "r2": r**2,
        "p": p,
        "stderr": stderr
    }