import math
import numpy as np
import pandas as pd
from scipy.stats import norm, binom

def calculate_power(sample_size, baseline, min_effect, test_sides, significance_level):
    estimatedTreatmentConversion = baseline + min_effect
    pooled_proportion = (baseline + estimatedTreatmentConversion) / 2

    # Standard error
    se = math.sqrt(
        (pooled_proportion * (1 - pooled_proportion) / sample_size) * 2
    )

    # Z-score for significance level
    z_alpha = norm.ppf(1 - significance_level / test_sides)

    # Effect size in terms of standard deviations
    effect_size = (baseline - estimatedTreatmentConversion) / se

    # Z-score for power
    z_beta = z_alpha - effect_size

    # Calculate power
    power = 1 - norm.cdf(z_beta)
    return power

sample_size = 1000
baseline = 0.1
min_effect = 0.2
test_sides = 2
significance_level = 0.05

power = calculate_power(sample_size, baseline, min_effect, test_sides, significance_level)
print("Power:", power)
