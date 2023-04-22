""" 
This code generates an exponential distribution with a lambda parameter 
of 2 and a size of 100,000. Then, it takes 1,000 samples of size 50 from 
this distribution, calculates their means, and stores them in a list. 
Finally, it plots both the original exponential distribution and the 
distribution of sample means using seaborn's histplot function. The resulting 
plot of the sample means should resemble a normal distribution, demonstrating 
the Central Limit Theorem in action.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Define the parameters for the exponential distribution
lam = 2
size = 100000

# Generate the exponential distribution
data = np.random.exponential(scale=1 / lam, size=size)

# Draw samples and compute sample means
sample_size = 50
num_samples = 1000
sample_means = []

for _ in range(num_samples):
    sample = np.random.choice(data, size=sample_size)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Plot the original exponential distribution
sns.histplot(data, kde=True, color="blue", ax=ax1)
ax1.set_title("Exponential Distribution")
ax1.set_xlabel("Values")
ax1.set_ylabel("Frequency")

# Plot the distribution of sample means
sns.histplot(sample_means, kde=True, color="red", ax=ax2)
ax2.set_title("Distribution of Sample Means")
ax2.set_xlabel("Sample Means")
ax2.set_ylabel("Frequency")

plt.show()
