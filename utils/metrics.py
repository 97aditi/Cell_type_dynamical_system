import numpy as np
from sklearn.metrics import r2_score


def compute_average_r_squared(predicted_activity, true_activity):
    """
    predicted_activity (num_trials x time_steps x num_neurons): predicted neural activity
    true_activity (num_trials x time_steps x num_neurons): true neural activity
    """
    # Compute r-squared for each trial
    r_squared = np.zeros(len(predicted_activity))
    for i in range(len(predicted_activity)):
        predicted_activity_this_trial = predicted_activity[i]
        true_activity_this_trial = true_activity[i]
        # remove any rows with NaNs
        idx = np.where(~np.isnan(true_activity_this_trial).any(axis=1))[0]
        predicted_activity_this_trial = predicted_activity_this_trial[idx]
        true_activity_this_trial = true_activity_this_trial[idx]
        r_squared[i] = r2_score(true_activity_this_trial, predicted_activity_this_trial)

    # Compute average r-squared across trials
    average_r_squared = np.mean(r_squared)
    return average_r_squared


def compute_r_squared_given_condition(predicted_activity, true_activity, indices):
    """ compute R2 for average left and right trial activity """

    N = predicted_activity[0].shape[-1]
    T = max([len(predicted_activity[i]) for i in range(len(predicted_activity))] )

    # let's create matrices of predicted and actual data
    predicted_data_mat = np.zeros((len(predicted_activity), T, N))
    actual_data_mat = np.zeros((len(predicted_activity), T, N))
    predicted_data_mat[:] = np.nan
    actual_data_mat[:] = np.nan
    for trial in range(len(predicted_activity)):
        len_trial = len(predicted_activity[trial])
        predicted_data_mat[trial, :len_trial,:] = predicted_activity[trial]
        len_trial = len(true_activity[trial])
        actual_data_mat[trial, :len_trial,:] = true_activity[trial]

    # get average activity for each neuron
    r2_score_neurons = np.zeros((N, 2))
    indices_left = np.where(np.array(indices)==1)[0]
    indices_right = np.where(np.array(indices)==-1)[0]
    average_left_predicted_data = np.nanmean(predicted_data_mat[indices_left], axis=0)
    average_right_predicted_data = np.nanmean(predicted_data_mat[indices_right], axis=0)
    average_left_actual_data = np.nanmean(actual_data_mat[indices_left], axis=0)
    average_right_actual_data = np.nanmean(actual_data_mat[indices_right], axis=0)
    
    # for each neuron, compute r_squared
    for neuron in range(N):
        r2_left = r2_score(average_left_actual_data[:,neuron], average_left_predicted_data[:,neuron])
        r2_right = r2_score(average_right_actual_data[:,neuron], average_right_predicted_data[:,neuron])
        r2_score_neurons[neuron, 0] = r2_left
        r2_score_neurons[neuron, 1] = r2_right

    return r2_score_neurons, predicted_data_mat, actual_data_mat

