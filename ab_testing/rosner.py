import math
import statsmodels.stats.power as power
import statsmodels.stats.api as api
from scipy.stats import norm

# using values from example 10.30 (Rosner)
p1 = 0.0015
d = 0.2
p2 = round((1 + d) * p1, 4)
k = 1  # equal sample sizes
q1 = 1 - p1
q2 = 1 - p2

p_bar = (p1 + k * p2) / (1 + k)

q_bar = 1 - p_bar
z_alpha_0_25 = 1.96
z_beta_0_8 = 0.84

# sample size calculator formula
n = round(
    (
        math.sqrt((p_bar * q_bar) * 2) * z_alpha_0_25
        + math.sqrt(p1 * q1 + p2 * q2) * z_beta_0_8
    )
    ** 2
    / (p1 - p2) ** 2,
    0,
)

cohens_d = api.proportion_effectsize(
    p1, p2
)  # Calculating effect size based on our expected rates
# print('cohens_d', cohens_d)

statsm_sample = power.tt_ind_solve_power(
    effect_size=cohens_d,
    nobs1=None,
    ratio=1,
    alpha=0.05,
    power=0.8,
    alternative="two-sided",
)

# print('statsm_sample', statsm_sample)


def get_p_value(z_score, test_sides):
    if test_sides == 1:
        # for an upper tailed test the sign of the z-score matters since only positive values will allows us to reject the null hypothesis
        return 1 - norm.cdf(z_score)
    else:
        # for a two tailed test we need to take the absolute value of the z-score since we are interested in the magnitude of the z-score, not the direction
        return 2 * (1 - norm.cdf(abs(z_score)))

def getMinSampleSize(baseline, minEffect, testSides, significanceLevel, powerLevel):
    estimatedTreatmentConversion = baseline + minEffect
    averageConversion = (baseline + estimatedTreatmentConversion) / 2
    numerator1 = (
        math.sqrt(averageConversion * (1 - averageConversion) * 2)
        * norm.ppf[1 - significanceLevel / testSides]
    )
    numerator2 = (
        math.sqrt(
            baseline * (1 - baseline)
            + estimatedTreatmentConversion * (1 - estimatedTreatmentConversion)
        )
        * norm.ppf(powerLevel)
    )
    denominator = (baseline - estimatedTreatmentConversion) ** 2
    return math.round((numerator1 + numerator2) ** 2 / denominator)




baseline_conversion_rate = 0.0015
minimum_detectable_effect_relative = 0.2
estimated_treatment_conversion_rate = round(
    (1 + minimum_detectable_effect_relative) * baseline_conversion_rate, 4
)
average_conversion_rate = (
    baseline_conversion_rate + estimated_treatment_conversion_rate
) / 2

min_sample_size = round(
    (
        math.sqrt((average_conversion_rate * (1 - average_conversion_rate)) * 2)
        * z_alpha_0_25
        + math.sqrt(
            baseline_conversion_rate * (1 - baseline_conversion_rate)
            + estimated_treatment_conversion_rate
            * (1 - estimated_treatment_conversion_rate)
        )
        * z_beta_0_8
    )
    ** 2
    / (baseline_conversion_rate - estimated_treatment_conversion_rate) ** 2,
    0,
)
