import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Set the Seaborn style
sns.set(style="dark")

def plot_binomial_pmf_cdf(n, p):
    # create an array of calues from 0 to n + 1
    x = np.arange(0, n+1)

    # the probability mass function shows the probability of getting a certain number of successes
    pmf = binom.pmf(x, n, p)

    # the cumulative distribution function shows the probability of getting a number of successes up to a certain value
    cdf = binom.cdf(x, n, p)

    x_axis_min = n / 2 - 100
    x_axis_max = n / 2 + 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # seaborn plot of the probability mass function with a smoothed curve
    sns.lineplot(x=x, y=pmf, ax=ax1)
    ax1.set_title(f'Probability Mass Function for Binomial Distribution (n={n}, p={p})')
    ax1.set_xlabel('Number of successes')
    ax1.set_ylabel('Probability of seeing exactly this number of successes')
    ax1.set_xlim([x_axis_min, x_axis_max])
    # ax1.plot(x, pmf, 'bo', ms=8, label='binom pmf')
    ax1.vlines(x, 0, pmf, colors='b', lw=1, alpha=0.5)

    # seaborn plot of the cumulative distribution function
    sns.lineplot(x=x, y=cdf, ax=ax2)
    ax2.set_title(f'Cumulative Distribution Function for Binomial Distribution (n={n}, p={p})')
    ax2.set_xlim([x_axis_min, x_axis_max])
    ax2.set_xlabel('Number of successes')
    ax2.set_ylabel('Probability of seeing this number of successes or less')
    # ax2.plot(x, cdf, 'bo', ms=8, label='binom cdf')
    ax2.vlines(x, 0, cdf, colors='b', lw=1, alpha=0.5)

    plt.show()

plot_binomial_pmf_cdf(1000, 0.5)

