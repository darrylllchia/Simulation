import random
import pandas as pd
import math
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Calculate Euclidean distance
def distance(x,y):
    return math.sqrt((y[1]-x[1])**2+(y[0]-x[0])**2)

# Finding the closest availble driver/rider, depending on the dictionary passed through
def find_closest_available(coord, available):
    min_dist = 100000000
    min_t = 10000000
    closest = None
    for avail in available.keys():
        if len(available[avail]) == 1 or isinstance(available[avail][1],tuple):
            d = distance(coord, available[avail][0])
            if d < min_dist:
                min_dist = d
                closest = avail
        else:
            t = available[avail][1]
            if t < min_t:
                min_t = t
                closest = avail
    return closest

# calculate driver's base earning from the trip, not including petrol cost
def calculate_trip(origin, destination):
    return 3 + 2*distance(origin, destination)

def calculate_petrol(origin, destination):
    return 0.2*distance(origin, destination)

def generate_random(distribution, param):
    if distribution == 'exp':
        return random.expovariate(param)
    elif distribution == 'unif':
        return random.uniform(0,param)
    return -1

def generate_location():
    return (random.uniform(0,20),random.uniform(0,20))

def generate_trip_time(origin, destination):
    d = distance(origin, destination)
    return random.uniform(d/25,d/16)

# function to add tuple to list and return the list sorted by time, which will be the 2nd element (index 1),
# in ascending order
def add_to_list(element, lst):
    if len(lst) == 0:
        return [element]
    l = 0
    r = len(lst) - 1
    time = element[1]
    while l <= r:
        mid = (l+r)//2
        if time < lst[mid][1]:
            r = mid - 1
        else:
            l = mid + 1
    lst.insert(l,element)
    return lst

def chi_square_exponential_fit(rider_waiting_time, alpha=0.05, num_bins=10, lambda_ = None):
    """
    Perform a Chi-Square goodness-of-fit test for an Exponential distribution.

    Parameters:
    - rider_waiting_time: List or array of observed waiting times
    - alpha: Significance level (default=0.05)
    - num_bins: Number of bins to categorize the data

    Returns:
    - Chi-Square statistic, p-value, and conclusion
    """
    if lambda_ == None:# Step 1: Estimate λ using MLE (1 / mean)
        lambda_ = 1 / np.mean(rider_waiting_time)

    # Step 2: Create bins for observed frequencies
    min_val, max_val = np.min(rider_waiting_time), np.max(rider_waiting_time)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    observed_freq, _ = np.histogram(rider_waiting_time, bins=bins)

    # Step 3: Compute Expected Frequencies
    expected_probs = np.diff(stats.expon.cdf(bins, scale=1/lambda_))  # P(X in bin)
    expected_freq = expected_probs * len(rider_waiting_time)

    # Ensure no bins have expected values < 5
    mask = expected_freq >= 5  # Only keep bins that meet the rule
    observed_freq = observed_freq[mask]
    expected_freq = expected_freq[mask]

    # Step 4: Compute Chi-Square statistic
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Step 5: Compute degrees of freedom
    df = len(observed_freq) - 1 - 1  # (bins - 1 for fitting - 1 for MLE)

    # Step 6: Compute p-value
    p_value = 1 - stats.chi2.cdf(chi_square_stat, df)

    # Step 7: Conclusion
    result = "Fail to Reject H0 (Fits well)" if p_value > alpha else "Reject H0 (Does not fit well)"

    return chi_square_stat, p_value, result, lambda_

def chi_square_uniform_fit(ihopethisworks, alpha=0.05, num_bins=10, a = None, b = None):
    """
    Perform a Chi-Square goodness-of-fit test for a Uniform distribution.

    Parameters:
    - ihopethisworks: List or array of observed values
    - alpha: Significance level (default=0.05)
    - num_bins: Number of bins to categorize the data

    Returns:
    - Chi-Square statistic, p-value, and conclusion
    """

    if a == None:# Step 1: Estimate MLE parameters (min and max)
        a = np.min(ihopethisworks)
        b = np.max(ihopethisworks)

    # Step 2: Create bins for observed frequencies
    bins = np.linspace(a, b, num_bins + 1)
    observed_freq, _ = np.histogram(ihopethisworks, bins=bins)

    # Step 3: Compute Expected Frequencies
    expected_prob = 1 / num_bins  # Equal probability for each bin
    expected_freq = np.full(num_bins, expected_prob * len(ihopethisworks))  # Scale by sample size

    # Ensure all expected frequencies are reasonable (> 5 rule)
    mask = expected_freq >= 5  # Only keep bins that meet the rule
    observed_freq = observed_freq[mask]
    expected_freq = expected_freq[mask]

    # Step 4: Compute Chi-Square statistic
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)

    # Step 5: Compute degrees of freedom
    df = len(observed_freq) - 1 - 1  # (bins - 1 for fitting - 1 for MLE)

    # Step 6: Compute p-value
    p_value = 1 - stats.chi2.cdf(chi_square_stat, df)

    # Step 7: Conclusion
    result = "Fail to Reject H0 (Fits well)" if p_value > alpha else "Reject H0 (Does not fit well)"

    return chi_square_stat, p_value, result, a, b

def two_d_chisq_uniform(lst):
    grid = [0 for _ in range(400)]
    for location in lst:
        x = location[0]
        y = location[1]
        grid[10 * int(x) + int(y)] += 1
    expected = [len(lst)/400 for _ in range(400)]
    expected = np.array(expected)
    grid = np.array(grid)
    chi_square_stat = np.sum((grid - expected) ** 2 / expected)

    df = len(grid) - 1 - 1  # (bins - 1 for fitting - 1 for MLE)

    p_value = 1 - stats.chi2.cdf(chi_square_stat, df)

    alpha = 0.05

    print(f"p value: {p_value}")
    if p_value > alpha:
        print("Conclusion: Fail to Reject H0 (Fits well)")
    else:
        print("Conclusion: Reject H0 (Does not fit well)")

def chi_square_normal_fit(data, alpha = 0.05, bins = 10, mu = None, sigma = None):
    def mle_normal_params(data):
        """Estimate MLE parameters (mean and std deviation) for a normal distribution."""
        mu = np.mean(data)
        sigma = np.std(data, ddof=0)  # MLE estimate uses ddof=0
        return mu, sigma

    """Perform a Chi-Square Goodness-of-Fit test for normality."""
    n = len(data)

    if mu == None:
    # Estimate MLE parameters
        mu, sigma = mle_normal_params(data)

    # Generate bin edges based on normal distribution percentiles
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(data, percentiles)

    # Compute observed frequencies
    observed_freqs, _ = np.histogram(data, bins=bin_edges)

    # Compute expected frequencies
    expected_probs = np.diff(stats.norm.cdf(bin_edges, loc=mu, scale=sigma))  # Probabilities for each bin
    expected_freqs = expected_probs * n

    # Avoid bins with zero expected frequency (replace with small value to avoid division errors)
    expected_freqs[expected_freqs == 0] = 1e-10  

    # Compute Chi-Square test statistic
    chi_square_stat = np.sum((observed_freqs - expected_freqs) ** 2 / expected_freqs)

    # Degrees of freedom (bins - estimated parameters - 1)
    df = bins - 3  # -3 because we estimate 2 parameters (mu, sigma) and lose 1 degree for sum of probs = 1
    p_value = stats.chi2.sf(chi_square_stat, df)

    result = "Fail to Reject H0 (Fits well)" if p_value > alpha else "Reject H0 (Does not fit well)"

    return chi_square_stat, p_value, result, mu, sigma

def one_sample_t_test(data, population_mean, alpha=0.05):
    """
    Performs a one-sample t-test.
    
    H0: The sample mean is equal to the population mean.
    H1: The sample mean is different from the population mean.
    
    Parameters:
        data (array-like): Sample data
        population_mean (float): Hypothesized population mean
        alpha (float): Significance level (default 0.05)
    
    Returns:
        t_statistic, p_value, conclusion
    """
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # Sample standard deviation
    n = len(data)
    
    # Calculate t-statistic
    t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    
    # Get p-value (two-tailed test)
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
    
    # Conclusion
    if p_value < alpha:
        conclusion = "Reject the null hypothesis (H0). Significant difference found."
    else:
        conclusion = "Fail to reject the null hypothesis (H0). No significant difference."
    
    return t_statistic, p_value, conclusion, population_mean

def find_gini(simple_earnings):
    # Function to compute the Gini coefficient
    def gini(array):
        """Calculate the Gini coefficient of a numpy array."""
        array = np.sort(array)  # Sort the array
        n = array.shape[0]
        return (2.0 * np.sum((np.arange(1, n + 1) * array)) / (n * np.sum(array))) - (n + 1) / n
    # Compute Gini coefficient for the simple case
    gini_simple = gini(simple_earnings)

    # Compute Lorenz Curve for simple case
    sorted_simple_earnings = np.sort(simple_earnings)
    cum_simple_earnings = np.cumsum(sorted_simple_earnings) / sorted_simple_earnings.sum()
    cum_simple_pop = np.arange(1, len(sorted_simple_earnings) + 1) / len(sorted_simple_earnings)

    # Plot Lorenz Curve for simple example
    plt.figure(figsize=(6, 5))
    plt.plot(cum_simple_pop, cum_simple_earnings, label="Lorenz Curve", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Equality Line")
    plt.xlabel("Cumulative Population Share")
    plt.ylabel("Cumulative Earnings Share")
    plt.title(f"Simple Lorenz Curve (Gini Coefficient: {gini_simple:.3f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display the simpler Gini coefficient
    return gini_simple
