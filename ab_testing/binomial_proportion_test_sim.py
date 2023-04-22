import math
import numpy as np
from scipy.stats import norm, binom


def z_statistic_no_correction(
    control_conversion_rate, exposed_conversion_rate, control_sample, exposed_sample
):
    p1 = control_conversion_rate
    p2 = exposed_conversion_rate
    n1 = control_sample
    n2 = exposed_sample

    pooled_prop = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(pooled_prop * (1 - pooled_prop) * (1 / n1 + 1 / n2))

    return z


# Example usage:
z_no_correction = z_statistic_no_correction(0.3, 0.35, 100, 100)
print("Z-statistic without continuity correction:", z_no_correction)

def z_statistic_with_correction(
    control_conversion_rate, exposed_conversion_rate, control_sample, exposed_sample
):
    p1 = control_conversion_rate
    p2 = exposed_conversion_rate
    n1 = control_sample
    n2 = exposed_sample

    pooled_prop = (p1 * n1 + p2 * n2) / (n1 + n2)
    continuity_correction = 1 / (2 * n1) + 1 / (2 * n2)
    z = (abs(p1 - p2) - continuity_correction) / math.sqrt(
        pooled_prop * (1 - pooled_prop) * (1 / n1 + 1 / n2)
    )

    return z


# Example usage:
z_with_correction = z_statistic_with_correction(0.3, 0.35, 100, 100)
print("Z-statistic with continuity correction:", z_with_correction)

# Calculate p-values from z-statistics
p_value_no_correction = 2 * (1 - norm.cdf(abs(z_no_correction)))
p_value_with_correction = 2 * (1 - norm.cdf(abs(z_with_correction)))

print("P-value without continuity correction:", p_value_no_correction)
print("P-value with continuity correction:", p_value_with_correction)

# Function to simulate the experiments and calculate the p-values
def simulate_experiments(num_experiments, n1, n2, true_proportion):
    p_values_no_correction = []
    p_values_with_correction = []
    
    for _ in range(num_experiments):
        # Simulate data for control and exposed groups
        control_conversions = binom.rvs(n1, true_proportion)
        exposed_conversions = binom.rvs(n2, true_proportion)

        # Calculate conversion rates
        control_conversion_rate = control_conversions / n1
        exposed_conversion_rate = exposed_conversions / n2

        # Calculate z-statistics without and with continuity correction
        z_no_correction = z_statistic_no_correction(control_conversion_rate, exposed_conversion_rate, n1, n2)
        z_with_correction = z_statistic_with_correction(control_conversion_rate, exposed_conversion_rate, n1, n2)

        # Calculate p-values from z-statistics
        p_value_no_correction = 2 * (1 - norm.cdf(abs(z_no_correction)))
        p_value_with_correction = 2 * (1 - norm.cdf(abs(z_with_correction)))

        p_values_no_correction.append(p_value_no_correction)
        p_values_with_correction.append(p_value_with_correction)

    return p_values_no_correction, p_values_with_correction


# Set parameters for the simulation
num_experiments = 1000
n1 = 10000
n2 = 10000
true_proportion = 0.3

# Run the simulation
p_values_no_correction, p_values_with_correction = simulate_experiments(num_experiments, n1, n2, true_proportion)

# Calculate the proportion of p-values less than 0.05
alpha = 0.05
prop_less_than_alpha_no_correction = np.mean(np.array(p_values_no_correction) < alpha)
prop_less_than_alpha_with_correction = np.mean(np.array(p_values_with_correction) < alpha)

print("Proportion of p-values less than 0.05 without continuity correction:", prop_less_than_alpha_no_correction)
print("Proportion of p-values less than 0.05 with continuity correction:", prop_less_than_alpha_with_correction)

