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


def demonstrate_peeking_problem(
    p1,
    p2,
    alpha,
    num_experiments,
    total_sample,
    peeks_count,
):
    p_values = np.empty(peeks_count).reshape(1, peeks_count)
    for _ in range(num_experiments):
        control_conversions = np.empty(0)
        exposed_conversions = np.empty(0)
        daily_p_values = np.empty(0)

        for _ in range(peeks_count):
            control_daily_conversions = binom.rvs(
                n=1, p=p1, size=total_sample // peeks_count
            )
            exposed_daily_conversions = binom.rvs(
                n=1, p=p2, size=total_sample // peeks_count
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


total_sample = 1000
peeks_count = 10
# Both groups have the same conversion rate, so there's no true effect
conversion_rate_a = 0.5
conversion_rate_b = 0.5
alpha = 0.05
num_experiments = 1000

p_values_frame = demonstrate_peeking_problem(
    conversion_rate_a,
    conversion_rate_b,
    alpha,
    num_experiments,
    total_sample,
    peeks_count,
)

significant_frame = p_values_frame.applymap(lambda x: x < alpha)

daily_frame= significant_frame.copy()
daily_frame["any_false_positive"] = daily_frame.any(axis=1)
daily_frame.loc["percent_false_positive"] = daily_frame.mean(axis=0)
print(daily_frame.loc["percent_false_positive"])

total_frame = significant_frame.copy()
cumulative_sum = total_frame.cumsum(axis=1)
# replace any values greater than 1 with 1
cumulative_sum = cumulative_sum.clip(
    0, 1
)  # update the dataframe with the updated values
total_frame[cumulative_sum == 1] = True
total_frame.loc["percent_false_positive"] = total_frame.mean(axis=0)

print(total_frame.loc["percent_false_positive"])


