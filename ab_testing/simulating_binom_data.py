import numpy as np
from scipy.stats import norm, binom

n1 = 200
n2 = 200
p1 = 0.4
p2 = 0.4

random_state = np.random.RandomState(2)

# this approach to creating simulated test data returns a success / failure value for each experiment unit (person)
data_A = binom.rvs(n=1, p=p1, size=n1, random_state=random_state)
# print('data_A:', data_A)
data_B = binom.rvs(n=1, p=p2, size=n2, random_state=random_state)
conversions_A = data_A.sum()
print("conversions_A:", conversions_A)
conversion_rate_A = data_A.mean()
conversion_rate_B = data_B.mean()

print("Conversion rate for A:", conversion_rate_A)
print("Conversion rate for B:", conversion_rate_B)

# this approach to creating simulated test data returns a total number of successes for the experiment, without the unit level data
control_conversions = binom.rvs(n1, p1, random_state=random_state)
print("control_conversions:", control_conversions)
exposed_conversions = binom.rvs(n2, p2, random_state=random_state)
control_conversion_rate = control_conversions / n1
exposed_conversion_rate = exposed_conversions / n2
print("Conversion rate for control:", control_conversion_rate)
print("Conversion rate for exposed:", exposed_conversion_rate)
