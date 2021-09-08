import matplotlib.pyplot as plt
import numpy as np
import csv
from measurement_mean_lifetime import data_simulation as simulate
from measurement_mean_lifetime import tau_estimator as estimate


# Properties of the experiment
total_decays = 113809
reading_intervals = np.array([i for i in np.arange(0, 61)])  # days
true_half_life = 73.81  # days
true_tau = true_half_life/np.log(2)  # days
# Get the middle of the time bins
times = 0.5*(reading_intervals[1:] + reading_intervals[:-1])


def read_file(filename):
    """
    Read data from a csv file to an array. Each row in the file becomes a row
    in the array
    """
    # Open the file
    with open(filename, 'r') as file:
        data = []
        # Define a csv reader
        rows = csv.reader(file, delimiter=',')
        for row in rows:
            # Convert the entries from string to float
            row_new = [float(x) for x in row]
            data.append(row_new)

        data = np.array(data)

        return data


def write_data_to_file(data, filename):
    """
    Write data from an arrow to a csv file. Each row of the array is written
    as a row in the file.
    """
    # Open the file
    with open(filename, 'w') as file:
        # Define a csv writer
        writer = csv.writer(file, delimiter=',')
        # Write to file
        for row in data:
            writer.writerow(row)


def repeated_simulation_file(repetitions, filename):
    """
    Repeat the experimental data simulation multiple times and save the
    data to a csv file
    """
    # Generate the simulated data sets
    data = []
    for i in range(repetitions):
        decays, errors = simulate(true_tau, total_decays, reading_intervals)
        data.append(decays)

    # Write the data to the file
    write_data_to_file(data, filename)


def many_taus(filename, tau_filename='taus.csv', save=False, accuracy=0.1):
    """
    Read the data contained in the file corresponding to simulated measurement
    data sets and estimate the mean life time for each data set.
    Save output to a file
    """
    # Read cumulative decay data from file
    data = read_file(filename)
    taus = []
    errors = []

    # Compute the mean life time for each data set
    for data_set in data:
        tau, error = estimate(data_set, reading_intervals)
        taus.append(tau)
        errors.append(error)

    # Save the list of computed taus and corresponding errors to a file
    # Each row is: tau, lower_error, upper_error
    if save:
        n = len(taus)
        zipped = [[taus[i], errors[i][0], errors[i][1]] for i in range(n)]
        write_data_to_file(zipped, tau_filename)

    return taus


def tau_distribution(filename='taus.csv'):
    # Read the mean life times from the file
    file_contents = read_file(filename)
    taus = file_contents[:, 0]

    plt.figure()
    plt.xlabel(r"$\tau$ [d]")
    plt.ylabel("Counts")
    hist_values, bins, patches = plt.hist(taus, bins=20,
                                          edgecolor="xkcd:darkblue")

    plt.savefig("tau_distribution.pdf")

    tau_mean = np.mean(taus)
    tau_std = np.std(taus, ddof=1)

    print("Estimate for the mean lifetime tau_hat = ({:.1f} +/- {:.1f})d".format(
            tau_mean, tau_std))

    return tau_mean, tau_std
