import math
import numpy as np
from scipy.stats import binom, norm

""" # Set the parameters for the A/B test 
n_A = 1000  # number of trials in group A
p_A = 0.2   # probability of success in group A
n_B = 1000  # number of trials in group B
p_B = 0.2  # probability of success in group B

# Generate the simulated data
random_state = np.random.default_rng(seed=13)  # set a random seed for reproducibility
data_A = binom.rvs(n=1, p=p_A, size=n_A, random_state=random_state)
data_B = binom.rvs(n=1, p=p_B, size=n_B, random_state=random_state)

# Calculate the conversion rates
conversion_rate_A = data_A.mean()
conversion_rate_B = data_B.mean()

print("Conversion rate for group A:", conversion_rate_A)
print("Conversion rate for group B:", conversion_rate_B)

# Calculate the number of successes in each group
success_A = np.sum(data_A)
success_B = np.sum(data_B)

# Calculate the pooled probability
pooled_prob = (success_A + success_B) / (n_A + n_B)

# Calculate the Z-score
z_score = (conversion_rate_B - conversion_rate_A) / np.sqrt(pooled_prob * (1 - pooled_prob) * (1/n_A + 1/n_B))

# Calculate the p-value
p_value = 1 - norm.cdf(z_score)

print("Z-score:", z_score)
print("p-value:", p_value) """


def getTestValue(
    controlConversionRate,
    exposedConversionRate,
    controlSample,
    exposedSample,
):
    averageConversionRate = (
        controlConversionRate * controlSample + exposedConversionRate * exposedSample
    ) / (controlSample + exposedSample)
    print("averageConversionRate:", averageConversionRate)

    return (
        abs(controlConversionRate - exposedConversionRate)
        - (1 / (2 * controlSample) + 1 / (2 * exposedSample))
    ) / np.sqrt(
        averageConversionRate
        * (1 - averageConversionRate)
        * (1 / controlSample + 1 / exposedSample)
    )


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


def run_experiment(n_A, n_B, p_A, p_B):
    # random_state = np.random.default_rng()
    data_A = binom.rvs(n=1, p=p_A, size=n_A)
    data_B = binom.rvs(n=1, p=p_B, size=n_B)
    conversion_rate_A = data_A.mean()
    conversion_rate_B = data_B.mean()

    z_score = z_statistic_no_correction(
        conversion_rate_B, conversion_rate_A, n_B, n_A
    )
    p_value = 1 - norm.cdf(z_score)

    return p_value


# Set parameters
n_A = 1000
n_B = 1000
p = 0.1
num_experiments = 10000

# Run multiple experiments
p_values = [run_experiment(n_A, n_B, p, p) for _ in range(num_experiments)]

# Calculate the proportion of p-values less than 0.05
false_positive_rate = sum(pv < 0.05 for pv in p_values) / num_experiments

print("Proportion of p-values less than 0.05:", false_positive_rate)
