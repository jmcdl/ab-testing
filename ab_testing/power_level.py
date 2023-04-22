import math
import numpy as np
import pandas as pd
from scipy.stats import norm, binom

pd.set_option("display.float_format", lambda x: "%.5f" % x)


def get_z_score(
    control_conversion_rate, exposed_conversion_rate, control_sample, exposed_sample
):
    p1 = control_conversion_rate
    p2 = exposed_conversion_rate
    n1 = control_sample
    n2 = exposed_sample

    pooled_prop = (p1 * n1 + p2 * n2) / (n1 + n2)
    z = (p2 - p1) / math.sqrt(pooled_prop * (1 - pooled_prop) * (1 / n1 + 1 / n2))

    return z


def get_p_value(z_score, test_sides):
    if test_sides == 1:
        # for an upper tailed test the sign of the z-score matters since only positive values will allows us to reject the null hypothesis
        return 1 - norm.cdf(z_score)
    else:
        # for a two tailed test we need to take the absolute value of the z-score since we are interested in the magnitude of the z-score, not the direction
        return 2 * (1 - norm.cdf(abs(z_score)))


def demonstrate_power(
    p1,
    p2,
    num_experiments,
    total_sample,
    division_count,
):
    p_values = np.empty(division_count).reshape(1, division_count)
    for _ in range(num_experiments):
        control_conversions = np.empty(0)
        exposed_conversions = np.empty(0)
        daily_p_values = np.empty(0)

        for _ in range(division_count):
            control_daily_conversions = binom.rvs(
                n=1, p=p1, size=total_sample // division_count
            )
            exposed_daily_conversions = binom.rvs(
                n=1, p=p2, size=total_sample // division_count
            )
            control_conversions = np.append(
                control_conversions, control_daily_conversions
            )
            exposed_conversions = np.append(
                exposed_conversions, exposed_daily_conversions
            )
            control_conversion_rate = control_conversions.mean()
            exposed_conversion_rate = exposed_conversions.mean()

            z = get_z_score(
                control_conversion_rate,
                exposed_conversion_rate,
                len(control_conversions),
                len(exposed_conversions),
            )

            pv = get_p_value(z, 2)
            daily_p_values = np.append(daily_p_values, pv)

        p_values = np.vstack((p_values, daily_p_values))

    p_values_frame = pd.DataFrame(p_values)

    return p_values_frame


total_sample = 2000
division_count = 20
conversion_rate_a = 0.45
conversion_rate_b = 0.5
min_detectable_effect = conversion_rate_b - conversion_rate_a
alpha = 0.05
test_sides = 2
num_experiments = 1000

p_values_frame = demonstrate_power(
    conversion_rate_a,
    conversion_rate_b,
    num_experiments,
    total_sample,
    division_count,
)

step = total_sample // division_count

significant_frame = p_values_frame.applymap(lambda x: x < alpha)

daily_frame = significant_frame.copy()
new_col_headers = np.arange(step, total_sample + step, step)
daily_frame = daily_frame.set_axis(new_col_headers, axis=1)
daily_frame.loc["percent_true_positive"] = daily_frame.mean(axis=0)
print(daily_frame.loc["percent_true_positive"])


def calculate_sample_size(
    baseline, min_detectable_effect, test_sides, significance_level, power_level
):
    p1 = baseline
    p2 = baseline + min_detectable_effect
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)

    numerator_part_1 = math.sqrt(p_bar * q_bar * 2) * norm.ppf(
        1 - significance_level / test_sides
    )
    numerator_part_2 = math.sqrt(p1 * q1 + p2 * q2) * norm.ppf(power_level)
    numerator = (numerator_part_1 + numerator_part_2) ** 2
    denominator = delta**2

    return round(numerator / denominator)


print(
    calculate_sample_size(
        conversion_rate_a, min_detectable_effect, test_sides, alpha, power_level=0.5
    )
)


def calculate_power(
    sample_size, baseline, min_detectable_effect, test_sides, significance_level
):
    p1 = baseline
    p2 = baseline + min_detectable_effect
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    q1 = 1 - p1
    q2 = 1 - p2
    delta = abs(p2 - p1)
    z_score_for_sig_level = norm.ppf(1 - significance_level / test_sides)

    part_1 = delta / math.sqrt(p1 * q1 / sample_size + p2 * q2 / sample_size)
    part_2a = math.sqrt(p_bar * q_bar * (2 / sample_size))
    part_2b = math.sqrt(p1 * q1 / sample_size + p2 * q2 / sample_size)
    part_2 = z_score_for_sig_level * part_2a / part_2b

    return norm.cdf(part_1 - part_2)


power = calculate_power(
    1000, conversion_rate_a, min_detectable_effect, test_sides, alpha
)
print(power)
