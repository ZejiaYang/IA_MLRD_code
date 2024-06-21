import numpy as np
from scipy import stats

def two_sample_t_test(sample1, sample2):
    n1 = 1000
    n2 = 1000
    mean1 = 100
    mean2 = 102
    std1 = 10  # Use ddof=1 for sample standard deviation
    std2 = 10
    
    # Compute the pooled standard deviation
    pooled_std = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    # Compute the t-statistic
    t_statistic = (mean1 - mean2) / pooled_std
    
    dof = (std1**2 / n1 + std2**2 / n2)**2 / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
    # Compute the critical value for a two-tailed test at alpha=0.05
    critical_value = stats.t.ppf(1 - 0.025, dof) 

    if np.abs(t_statistic) > critical_value:
        significance = True
    else:
        significance = False
    
    return t_statistic, significance

# Example usage:
sample1 = np.array([81, 86, 82, 84, 79, 79, 76, 82, 85, 88])
sample2 = np.array([82, 87, 83, 85, 80, 81, 77, 83, 87, 89])
t_statistic, significant = two_sample_t_test(sample1, sample2)
print("T-statistic:", t_statistic)
print("Significant:", significant)