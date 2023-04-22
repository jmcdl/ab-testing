import numpy as np
from scipy.stats import binom
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Define the parameters for the binomial distribution
n_trials = 100
p_success = 0.5
size = 100000

# Generate the binomial distribution
data = np.random.binomial(n_trials, p_success, size)
print(len(data))

pmf = binom.rvs(n_trials, p_success, size=size)
print(len(pmf))

# Draw samples and compute sample means
sample_size = 50
num_samples = 1000
sample_means = []

for _ in range(num_samples):
    sample = np.random.choice(data, size=sample_size)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)

# Plot the original binomial distribution
plt.figure()
sns.histplot(data, kde=True, discrete=True, color='blue')
plt.title('Binomial Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Plot the distribution of sample means
plt.figure()
sns.histplot(sample_means, kde=True, color='red')
plt.title('Distribution of Sample Means')
plt.xlabel('Sample Means')
plt.ylabel('Frequency')

# plt.show()
