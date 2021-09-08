# -*- coding: utf-8 -*-
"""
Data Analysis, Fall Semester 2018

Group Project 1: Characterization of a radioactive source, Task 3

Authors: Stefanie Jucker, Andrej Maraffio, Mirko Mirosavljevic, 
Manuela Rohrbach, Stefan von Rohr
"""
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon


# Properties of the experiment
total_decays = 113809
reading_intervals = np.array([i for i in np.arange(0, 61)])  # days
true_half_life = 73.81  # days
true_tau = true_half_life/np.log(2)  # days
# Get the middle of the time bins
times = 0.5*(reading_intervals[1:] + reading_intervals[:-1])


def pull_function(true_tau, tau_hat, sig_tau_hat):
    return (tau_hat - true_tau)/sig_tau_hat


#def limited_lifetime_pdf(t, tau, t_max=reading_intervals[-1]):
#    """
#    Probability density function for exponential decay when you measure for
#    a certain time, which limits the possible measured lifetimes
#    """
#    pdf = expon.pdf(t, scale=tau)/expon.cdf(t_max, scale=tau)
#
#    return pdf
#

def limited_lifetime_cdf(t, tau, t_max=reading_intervals[-1]):
    """
    Cumulative density function for exponential decay when you measure for
    a certain time, which limits the possible measured lifetimes
    """
    cdf = expon.cdf(t, scale=tau)/expon.cdf(t_max, scale=tau)

    return cdf


def inv_limited_lifetime_cdf(u, tau=true_tau, t_end=60):
    """
    Inverse function of the cdf of an exponential with limitied time.
    Finds which t has u = cdf*(t)
    """
    k = expon.cdf(t_end, scale=tau)
    t = -tau*np.log(1-u*k)

    return t


def data_simulation(true_tau, total_decays, reading_intervals):
    """
    Simulate a dataset for the measured decays of the radioactive source

    Parameters
    ----------
    true_tau: float, known mean life time of the source (in days)
    total_decays: int>0, total number of decays to simulate
    reading_intervals: list, boundaries of reading intervals (in days)

    Returns
    -------
    decays: list of int>0, readings of number of decays at the given times
    errors: list of floats, error on decays
    """
    random_uniform = np.random.sample(total_decays)
    lifetimes = inv_limited_lifetime_cdf(random_uniform, true_tau)

    # Count how many decays are in the asked measuring intervals
    # -> Histogram the data in a cumulative way
    decays_per_interval = np.histogram(lifetimes, bins=reading_intervals)[0]

    decays = []
    for i in range(1, len(decays_per_interval) + 1):
        value = sum(decays_per_interval[:i])
        decays.append(value)

    errors = np.sqrt(decays)

    return decays, errors


def counts_per_day(cumulative_counts):
    """
    Reduce data to decay counts per day from the cumulative experimental data.
    """
    # Get difference between consecutive entries
    reduced_counts = np.diff(cumulative_counts)

    # Add the first measurement unaltered
    reduced_counts = np.insert(reduced_counts, 0, cumulative_counts[0])

    return reduced_counts


def binned_ll(tau_range, hist_values, bin_edges):
    """
    Computes the binned log likelihood for a range of mean lifetimes (tau_range)
    """
    # Total entries in all bins
    total_events = sum(hist_values)

    # Constant addition term for the log likelihood
    const = sum([1/factorial(n) for n in hist_values])

    ll = []

    for tau in tau_range:
        # Expected number of entries in each bin
        vs = total_events*(limited_lifetime_cdf(bin_edges[1:], tau)
                           - limited_lifetime_cdf(bin_edges[:-1], tau))
        summands = hist_values*np.log(vs) - vs
        ll.append(sum(summands))

    # Convert to a numpy array
    ll = np.array(ll)

    # Add the constant term to all list entries
    ll += const

    return ll


def tau_estimator(decays_cumulative, reading_intervals, accuracy=0.1):
    """
    Compute an estimate tau hat for the mean lifetime using a binned
    log likelihood method
    """
    # Get the measured decays per day (not cumulative)
    decays_per_day = counts_per_day(decays_cumulative)

    # Define a range of possible taus and compute the binned log-likelihood
    tau_range = np.arange(100, 115, accuracy)
    log_likelihood = binned_ll(tau_range, decays_per_day, reading_intervals)

    # Determine the estimator through the maximum of the log likelihood
    max_index = np.argmax(log_likelihood)
    tau_hat = tau_range[max_index]

    # Determine an error on the estimate
    log_likelihood -= max(log_likelihood)
    tau_e = tau_range[log_likelihood > -0.5]
    sig_tau_hat = (tau_e.max() - tau_hat, tau_hat - tau_e.min())

    return tau_hat, sig_tau_hat


def fitted_plot(reading_intervals, decays_cumulative, errors, times, tau_hat):
    # Create the best fit line on the cumulative data
    fit_n = total_decays*limited_lifetime_cdf(reading_intervals[1:], tau_hat)

    # Process data and fit line for declining plots
    decays_sliced = counts_per_day(decays_cumulative)
    errors_sliced = np.sqrt(decays_sliced)
    fit_sliced = counts_per_day(fit_n)

    # Plot of cumulative data set (our measured data)
    plt.figure()
    plt.errorbar(times, decays_cumulative, yerr=errors, ecolor="black",
                 fmt='b.', ms=5, mew=0.5, label="Simulated measurement data")
    plt.xlabel("time [d]")
    plt.ylabel("number of registered decays")
    plt.xlim(0, reading_intervals[-1] + 2)
    plt.ylim(0, total_decays+5000)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.savefig("decay_measurement_cumulative.pdf")

    # Plot of declining data set with fitted line
    plt.figure()
    plt.plot(reading_intervals[1:], fit_sliced, color='orange', label="fit line", zorder=1)
    plt.errorbar(reading_intervals[1:], decays_sliced, yerr=errors_sliced, fmt='b.', linestyle="None", 
                 label="Simulated measurement data", zorder=2)
    plt.xlabel("time [d]")
    plt.ylabel("number of registered decays per day")
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.savefig("decay_measurement_declining_fitted.pdf")


def pull_calc(repetitions):
    pull = []
    
    for i in range(repetitions):
        # Generated simulated measurement data again
        decays_cumulative, errors = data_simulation(true_tau, total_decays,
                                                    reading_intervals)
    
        # Estimate the mean lifetime again
        tau_hat, sig_tau_hat = tau_estimator(decays_cumulative, reading_intervals)
        
        # calculating the pull again
        sig_tau_hat_average = max(sig_tau_hat)
        pull.append((tau_hat - true_tau)/sig_tau_hat_average)
    
    plt.figure()
    plt.hist(pull, bins=30, edgecolor="xkcd:darkblue", range=(-3,3))
    plt.xlabel("pull")
    plt.ylabel("counts")
    plt.xlim(-3,3)
    plt.savefig('pull.pdf')
    
    return pull


def task_3():
    print("Actual mean lifetime:    {:.1f}d".format(true_tau))

    # Generated simulated measurement data
    decays_cumulative, errors = data_simulation(true_tau, total_decays,
                                                reading_intervals)

    # Estimate the mean lifetime
    tau_hat, sig_tau_hat = tau_estimator(decays_cumulative, reading_intervals)

    print("Estimated mean lifetime: ({:.1f} + {:.1f} / - {:.1f})d".format(
        tau_hat, sig_tau_hat[0], sig_tau_hat[1]))

    # Create a plot with a maximum likelihood fit
    fitted_plot(reading_intervals, decays_cumulative, errors, times, tau_hat)

    # calculating the pull
    pull = pull_calc(1000)
#    print("Pull of this meausurement:", pull[0])

    print("mean of the pull:", np.mean(pull))
    print("standard deviation of the pull:", np.sqrt(np.cov(pull)))
#    print(pull)


if __name__ == "__main__":
    # Task 3: Measurement of mean lifetime
    task_3()
