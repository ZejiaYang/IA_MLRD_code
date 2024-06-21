from scipy import stats
import numpy as np

# Sample data
sample1 = np.array([81, 86, 82, 84, 79, 79, 76, 82, 85, 88])
sample2 = np.array([82, 87, 83, 85, 80, 81, 77, 83, 87, 89])

# Paired t-test
t_statistic, p_value = stats.ttest_rel(sample1, sample2)

print("T-statistic:", t_statistic)
print("P-value:", p_value)
