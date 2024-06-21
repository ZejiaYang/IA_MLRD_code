import numpy as np
from scipy import stats

def two_sample_t_test(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    std1 = np.std(sample1, ddof=1)  
    std2 = np.std(sample2, ddof=1)
    
    pooled_std = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    
    t_statistic = (mean1 - mean2) / pooled_std
    
    dof = (std1**2 / n1 + std2**2 / n2)**2 / ((std1**2 / n1)**2 / (n1 - 1) + (std2**2 / n2)**2 / (n2 - 1))
    
    # Compute the critical value for a two-tailed test at alpha=0.05
    critical_value = stats.t.ppf(1 - 0.025, dof)  # 0.025 because it's a two-tailed test
    # Determine statistical significance
    if np.abs(t_statistic) > critical_value:
        significance = True
    else:
        significance = False
    
    return t_statistic, significance


sample1 = np.array([81, 86, 82, 84, 79, 79, 76, 82, 85, 88])
sample2 = np.array([82, 87, 83, 85, 80, 81, 77, 83, 87, 89])
t_statistic, significant = two_sample_t_test(sample1, sample2)
print("T-statistic:", t_statistic)
print("Significant:", significant)
