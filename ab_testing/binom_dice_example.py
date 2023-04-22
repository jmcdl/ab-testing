import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn style
sns.set(style="dark")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# roll a die 20 times and determine the probability of rolling the same value
n = 20
p = round(1/6, 3)

x= np.arange(0, n+1)
# the probability mass function shows the probability of getting a certain number of successes
pmf = binom.pmf(x, n, p)

# the cumulative distribution function shows the probability of getting a number of successes up to a certain value
cdf = binom.cdf(x, n, p)

# seaborn plot of the probability mass function with a smoothed curve
sns.lineplot(x=x, y=pmf, ax=ax1)
ax1.set_title(f'Probability Mass Function for Binomial Distribution (n={n}, p={p})')
ax1.set_xlabel('Number of successes')
ax1.set_ylabel('Probability')
# ax1.plot(x, pmf, 'bo', ms=8, label='binom pmf')
# ax1.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)
ax1.set_xlim([3,9])
# ax1.set_xticks(x)
# Add labels to the points
# for i in range(len(pmf)):
#     if round(pmf[i], 3) > 0.000: 
#         ax1.text(x[i], pmf[i], "  " + str(round(pmf[i], 3)), ha='left', va='bottom', fontsize=10)


# seaborn plot of the cumulative distribution function
sns.lineplot(x=x, y=cdf, ax=ax2)
ax2.set_title(f'Cumulative Distribution Function for Binomial Distribution (n={n}, p={p})')
ax2.set_xlabel('Number of successes')
ax2.set_ylabel('Probability')
# ax2.plot(x, cdf, 'bo', ms=8, label='binom cdf')
# ax2.vlines(x, 0, cdf, colors='b', lw=5, alpha=0.5)
ax2.set_xticks(x)
# Add labels to the points
for i in range(len(cdf)):
    if round(cdf[i], 3) < 1: 
        ax2.text(x[i], cdf[i], "  " + str(round(cdf[i], 3)), ha='left', va='bottom', fontsize=10)

plt.show()
