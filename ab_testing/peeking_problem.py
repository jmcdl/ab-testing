import math
import numpy as np
from scipy.stats import norm, binom


def run_experiment(p1, p2, n1, n2):
    control_conversions = binom.rvs(n1, p1)
    exposed_conversions = binom.rvs(
        n2,
        p2,
    )
    control_conversion_rate = control_conversions / n1
    exposed_conversion_rate = exposed_conversions / n2
    return control_conversion_rate, exposed_conversion_rate


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
    peeking_points,
):
    false_positives = 0
    print("peeking_points", peeking_points)
    for _ in range(num_experiments):
        false_positive = False

        for peek_point in peeking_points:
            print("peek_point", peek_point)
            (control_conversion_rate, exposed_conversion_rate) = run_experiment(
                p1, p2, peek_point, peek_point
            )
            z = get_z_score(
                control_conversion_rate,
                exposed_conversion_rate,
                peek_point,
                peek_point,
            )
            pv = get_p_value(z, 2)
            print(pv)
            if pv < alpha:
                false_positive = True
                break

        if false_positive:
            false_positives += 1

    return false_positives / num_experiments


sample_size = 1000
peeks = 1
# Both groups have the same conversion rate, so there's no true effect
conversion_rate_a = 0.5
conversion_rate_b = 0.5
alpha = 0.05
num_experiments = 1000
peeking_points = np.arange(
    sample_size // peeks, sample_size + sample_size // peeks, sample_size // peeks
)

false_positive_rate = demonstrate_peeking_problem(
    conversion_rate_a,
    conversion_rate_b,
    alpha,
    num_experiments,
    peeking_points,
)
print(f"False positive rate with peeking: {false_positive_rate:.2%}")
