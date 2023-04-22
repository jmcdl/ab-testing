import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom


def plot_binomial_pmf(n, p, alpha, test_sides):
    # create an array of values from 0 to n + 1
    x = np.arange(0, n + 1)
    x_proportion = x / n
    proportion_variance = p * (1 - p) / n
    proportion_sd = math.sqrt(proportion_variance)

    # the probability mass function shows the probability of getting a certain number of successes
    pmf = binom.pmf(x, n, p)
    pmf_proportion = pmf / n

    x_standard_deviations = (x_proportion - p) / proportion_sd

    plt.figure(figsize=(20, 6))
    ax = sns.lineplot(x=x_standard_deviations, y=pmf_proportion)
    plt.title(
        f"Binomial Distribution (n={n}, p={p}), variance={proportion_variance:.5f}"
    )
    plt.xlabel("Standard Deviations from the Mean")
    plt.ylabel("Probability of seeing exactly this proportion of successes")
    plt.xlim([-4, 4])
    plt.vlines(x_standard_deviations, 0, pmf_proportion, colors="b", lw=1, alpha=0.5)

    if test_sides == "Two-sided":
        # Show the range of values for a 5% significance level for a two-sided test
        z_score = norm.ppf(1 - alpha / 2)
        lower_bound_sd = -z_score
        upper_bound_sd = z_score
        ax.fill_between(
            x_standard_deviations,
            0,
            pmf_proportion,
            where=(
                (x_standard_deviations <= lower_bound_sd)
                | (x_standard_deviations >= upper_bound_sd)
            ),
            color="r",
            alpha=0.3,
        )

    if test_sides == "One-sided":
        # Show the 5% significance level
        z_score = norm.ppf(1 - alpha)
        upper_bound_sd = z_score
        ax.fill_between(
            x_standard_deviations,
            0,
            pmf_proportion,
            where=((x_standard_deviations >= upper_bound_sd)),
            color="r",
            alpha=0.3,
        )

    plt.show()


plot_binomial_pmf(1000, 0.5, 0.05, "Two-sided")
