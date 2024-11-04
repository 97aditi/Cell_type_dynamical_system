import numpy as np
import pickle
import ssm
import os
from pathlib import Path

def load_model(region, N, r, M, seed, type):
    """ load an lds model with from fitted parameters"""
    if type == 'constrained':
        save_folder = '../results_fof_ads/'+str(region)+'_constrained_lds_N_'+str(N)+'_d_'+str(r)+'_seed_'+str(seed)+'/best_params.npy'
    elif type == 'unconstrained':
        save_folder = '../results_fof_ads/'+str(region)+'_unconstrained_lds_N_'+str(N)+'_d_'+str(r)+'_seed_'+str(seed)+'/best_params.npy'

    model_path = save_folder + '/models/model_trained.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_signs(path = '../data_for_rat_X062/fof_ads_data_non_smoothed.npz', region='both', n_fof=36):
    """ load the signs of the neurons"""
    data = np.load(path)
    cell_ids = data['cell_ids']
    if region == 'fof':
        cell_ids = cell_ids[:n_fof]
    elif region == 'ads':
        cell_ids = cell_ids[n_fof:]
    # let's get the signs of the neurons from a pickle fileneuer
    # remove everything after the last / from path
    path = path[:path.rfind('/')]
    path += '/signs_dict.pickle'
    with open(path, 'rb') as f:
        signs_dict = pickle.load(f)
    signs = []
    for cell_id in cell_ids:
        signs.append(signs_dict[cell_id])
    signs = np.array(signs)
    return signs
    

def load_data(path = 'data_for_rat_X062/fof_ads_data_non_smoothed.npz', region='both', n_fof=36):
    """ Load the neural data / inuts / choices and split into train and test sets"""

    data = np.load(path)
    # trials x time steps x neurons
    neural_data = data['neural_data']
    if region == 'fof':
        # only get data for FOF
        neural_data = neural_data[:, :, :n_fof]
    elif region == 'ads':
        # only get data for ADS
        neural_data = neural_data[:, :, n_fof:]
    clicks = data['clicks_inputs']
    correct_trials = data['choices']
    num_trials = neural_data.shape[0]
    N = neural_data.shape[2]

    # accuracy of the animal
    acc = np.mean(correct_trials)

    datas = []
    datas_test = []
    choice_train = []
    choice_test = []
    inputs = []
    inputs_test = []

    # to mean center data later on
    mean_activity_per_neuron = np.zeros(N)
    num_train_trials = 0

    for trial in range(num_trials):
        this_trial = neural_data[trial]
        # let's also find the indices of the non-nan rows
        idx = np.where(~np.isnan(neural_data[trial]).any(axis=1))[0]
        inputs_this_trial = clicks[trial]
        inputs_this_trial = inputs_this_trial[idx]
        # let's get the correct choice for this trial, 0 is left clicks, 1 is right clicks
        correct_choice = np.sign(np.sum(inputs_this_trial[:,0])-np.sum(inputs_this_trial[:,1]))
        animal_choice = correct_choice if correct_trials[trial] == 1 else -correct_choice
        if trial % 5 == 0:
            datas_test.append(this_trial[idx])
            inputs_test.append(inputs_this_trial)
            choice_test.append(animal_choice)
        else:
            choice_train.append(animal_choice)
            datas.append(this_trial[idx])
            inputs.append(inputs_this_trial)
            mean_activity_per_neuron += np.mean(this_trial[idx], axis=0)
            num_train_trials += 1

    mean_activity_per_neuron /= num_train_trials

    # mean center data
    for i in range(len(datas)):
        datas[i] -= mean_activity_per_neuron
    for i in range(len(datas_test)):
        datas_test[i] -= mean_activity_per_neuron
    return datas, datas_test, inputs, inputs_test, choice_train, choice_test, acc, mean_activity_per_neuron



def save_run(save_folder, model_trained=None, model_true=None, ep=None, **vars_to_save):
    """ function to save trained model"""
    save_folder = Path(save_folder)
    # check if the folder exists
    if not save_folder.exists():
        os.mkdir(save_folder)

    model_save_folder = save_folder / 'models'

    # save the trained model
    if model_trained is not None:
        if not model_save_folder.exists():
            os.mkdir(model_save_folder)

        if ep is not None:
            trained_model_save_path = model_save_folder / ('model_trained_' + str(ep) + '.pkl')
        else:
            trained_model_save_path = model_save_folder / 'model_trained.pkl'
        save_file = open(trained_model_save_path, 'wb')
        pickle.dump(model_trained, save_file)
        save_file.close()

    # save the true model, if it exists
    if model_true is not None:
        if not model_save_folder.exists():
            os.mkdir(model_save_folder)

        true_model_save_path = model_save_folder / 'model_true.pkl'
        model_true.save(path=true_model_save_path)

    for k, v in vars_to_save.items():
        save_path = save_folder / (k + '.pkl')

        save_file = open(save_path, 'wb')
        pickle.dump(v, save_file)
        save_file.close()