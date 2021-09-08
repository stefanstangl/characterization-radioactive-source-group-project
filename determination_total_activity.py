# -*- coding: utf-8 -*-
"""
Data Analysis, Fall Semester 2018

Group Project 1: Characterization of a radioactive source, Task 4

Authors: Stefanie Jucker, Andrej Maraffio, Mirko Mirosavljevic, Manuela Rohrbach, Stefan von Rohr
"""
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal
import numpy as np
from scipy.stats import uniform, poisson
from measurement_mean_lifetime import tau_estimator, data_simulation

# Properties of the experiment
true_half_life = 73.81  # days
true_tau = true_half_life/np.log(2)  # days
total_decays = 113809
reading_intervals = np.array([i for i in np.arange(0, 61)])  # days

#calculated in Task_3:
tau = 107.8  # days
tau_err = 2.0 # days
half_life = tau*np.log(2) # days
half_life_err = tau_err*np.log(2) # days

# Mean values for 2D Gaussian distribution
mu_x = 0  # cm
mu_y = 0  # cm

# Standard deviations for 2D Gaussian distribution
std_x = 3  # cm
std_y = 6  # cm

# Covariance matrix of uncorrelated x and y
cov_xy = np.array([[std_x**2, 0],
                  [0, std_y**2]])

# Position and dimensions of the detector
detector_width_x = 4  # cm
detector_width_y = 5  # cm
detector_center = (0, 0)


def detector_acceptance(mus, cov, width_x, width_y, center, number_of_points,
                        number_of_repetitions, plot=False, systematic_position_error=False):
    """
    Determine the geometric acceptance of the detector
    with a systematic uncertainty on the detector position if systematic_position_error=True
    """
    # Define the borders of the detector
    x_low = center[0] - 0.5*width_x
    x_high = center[0] + 0.5*width_x
    y_low = center[1] - 0.5*width_y
    y_high = center[1] + 0.5*width_y

    # Repetition of experiment multiple times to get a mean and std
    acceptances = []
    for i in range(number_of_repetitions):
        
        # If systematic error on position, generate uniform uncertainty on borders of detector
        if systematic_position_error:
            x_error = uniform.rvs(loc=-0.2, scale=0.4) # cm
            y_error = uniform.rvs(loc=-0.2, scale=0.4) # cm
            x_low += x_error
            x_high += x_error
            y_low += y_error
            y_high += y_error
        
        # Generate position of particles
        points = multivariate_normal(mus, cov, size=number_of_points)

        # Determine which points will hit the detector
        detected_points = 0
        for x, y in points:
            if (x_low <= x <= x_high) and (y_low <= y <= y_high):
                detected_points += 1

        acceptances.append(detected_points/number_of_points)

    acceptance = np.mean(acceptances)
    error = np.std(acceptances, ddof=1)

    if plot:
        heatmap(mus, cov, x_low, x_high, y_low, y_high)

    return acceptance, error


def heatmap(mus, cov, x_low, x_high, y_low, y_high, size=1000000):
    points = multivariate_normal(mus, cov, size=size)

    plt.figure()
    plt.ylabel("x position [cm]")
    plt.xlabel("y position [cm]")

#        # Plot the scattered points
#        plt.plot(points[:, 0], points[:, 1], 'b.', ms=2)

    # Make a heat map of the scattered points
    n_std = 2.5
    x_lim = [mus[0] - n_std*np.sqrt(cov[0, 0]),
             mus[0] + n_std*np.sqrt(cov[0, 0])]
    y_lim = [mus[1] - n_std*np.sqrt(cov[1, 1]),
             mus[1] + n_std*np.sqrt(cov[1, 1])]
    hist_values, x_edges, y_edges = np.histogram2d(points[:, 1],
                                                   points[:, 0],
                                                   bins=(100, 50),
                                                   range=[y_lim, x_lim])
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    plt.imshow(hist_values.T, extent=extent, origin='lower')
    plt.colorbar()

    # Plot the boundaries of the detector
    plt.plot(2*[y_low], [x_low, x_high], 'r-', linewidth=0.6)
    plt.plot(2*[y_high], [x_low, x_high], 'r-', linewidth=0.6)
    plt.plot([y_low, y_high], 2*[x_low], 'r-', linewidth=0.6)
    plt.plot([y_low, y_high], 2*[x_high], 'r-', linewidth=0.6)

    plt.savefig("Electron_scatter_heatmap.pdf")


def detector_activity(acceptance, tau=tau, total_decays=total_decays,
                      t_activity=1, t_end=60):
    """ Compute the activity of the source """
    N_0 = total_decays/(1-np.exp(-t_end/tau))
    C = N_0/(tau*24*60*60*acceptance)
    act = C*np.exp(-t_activity/tau)

    return act

def tau_total_decays_cov(true_tau, reading_intervals, reps):
    """ Finds the covariance of the total number of events and the mean life time"""
    # Generate total number of events poisson distributed
    tot_decay = poisson.rvs(total_decays, size=reps)
    
    # Determine mean life time for each of the total number of events
    tau_est = []
    for i in range(reps):
        data = data_simulation(true_tau, tot_decay[i], reading_intervals)[0]
        tau_est.append(tau_estimator(data, reading_intervals)[0])
    tau_est = np.array(tau_est)
    
    # Calculate the covariance of the mean life time and the total number of evenets
    return np.cov(tau_est, tot_decay)
    
def activity_error(acc, acc_error, total_decays, tau_hat, tau_error, cov_tau_N, no_days=60):
    """Calculates the error on the activity of the source"""
    a = acc
    t = no_days*24*60*60 # seconds
    T = tau_hat
    n = total_decays
    derivative_acc = n/((a**2)*T - (a**2)*np.exp(t/T)*T)
    derivative_tau = (n*(np.exp(t/T)*(t - T) + T))/(a*((-1 + np.exp(t/T))**2)*(T**3))
    derivative_total_decays = -(1/(a*T - a*np.exp(t/T)*T))
    jacobian = np.array([derivative_acc, derivative_tau, derivative_total_decays])
    cov_matrix = np.array([[acc_error**2, 0, 0],
                           [0, tau_error**2, cov_tau_N],
                           [0, cov_tau_N, total_decays]])
    return np.sqrt(np.dot(jacobian, np.dot(cov_matrix, jacobian.T)))

def activity_error_uncorrelated(acc, acc_error, total_decays, tau_hat, tau_error, no_days=60):
    """Calculates the error on the activity of the source"""
    a = acc
    t = no_days*24*60*60 # seconds
    T = tau_hat
    n = total_decays
    derivative_acc = n/((a**2)*T - (a**2)*np.exp(t/T)*T)
    derivative_tau = (n*(np.exp(t/T)*(t - T) + T))/(a*((-1 + np.exp(t/T))**2)*(T**3))
    derivative_total_decays = -(1/(a*T - a*np.exp(t/T)*T))
    jacobian = np.array([derivative_acc, derivative_tau, derivative_total_decays])
    cov_matrix = np.array([[acc_error**2, 0, 0],
                           [0, tau_error**2, 0],
                           [0, 0, total_decays]])
    return np.sqrt(np.dot(jacobian, np.dot(cov_matrix, jacobian.T)))

def task_4(ON):
    """ Compute task 4 without generating plots if ON=False"""
    # Calculated in Task_3:
    tau_s = tau*24*60*60  # seconds
    tau_err_s = tau_err*24*60*60 # seconds
    
    # Compute the acceptance of our detector and the 
    acc, acc_err = detector_acceptance([mu_x, mu_y], cov_xy, detector_width_x,
                                       detector_width_y, detector_center,
                                       10000, 1000, plot=ON)
    print("The geometric acceptance of the detector is {:.3f} +/- {:.3f}".format(
            acc, acc_err))

    activity = detector_activity(acc)
    
    repetitions = 1000
    
    cov_tau_N_mat = tau_total_decays_cov(true_tau, reading_intervals, repetitions) #days*number
    cov_tau_N = cov_tau_N_mat[0][1]*24*60*60 # seconds
    
    activity_err = activity_error(acc, acc_err, total_decays, tau_s, tau_err_s, cov_tau_N)
    
    print(f"The source's activity for t = 1d is {activity:.3} Bq")
    print(f'Error on activity: {activity_err} for {repetitions} repetitions')
    
    #Uncorrelated activity error
    uncor_act_err = activity_error_uncorrelated(acc, acc_err, total_decays, tau_s, tau_err_s)
    print(f'Gaussian error propagation: {uncor_act_err}')
    
    # Compute the acceptance of the detector 
    acc_syst, acc_syst_err = detector_acceptance([mu_x, mu_y], cov_xy, detector_width_x,
                                       detector_width_y, detector_center,
                                       10000, 1000, systematic_position_error=True)
    print("The geometric acceptance with a systematic error for the position of the detector is {:.3f} +/- {:.3f}".format(
            acc_syst, acc_syst_err))

    activity_syst = detector_activity(acc_syst)

    print("The source's activity for t = 1d is {:.3} Bq".format(activity_syst))
    # Same for acc_with_systematic
    
    activity_syst_err = activity_error(acc_syst, acc_syst_err, total_decays, tau_s, tau_err_s, cov_tau_N)
    print(f'Error on activity with systematic detector missalignement: {activity_syst_err} for {repetitions} repetitions')

if __name__ == "__main__":
    ON = True
    task_4(ON)