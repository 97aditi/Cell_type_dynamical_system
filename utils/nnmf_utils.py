import numpy as np 
import cvxpy as cp
from tqdm import tqdm
import os
from scipy.linalg import block_diag
from sklearn.decomposition import NMF
import argparse
import numpy.random as npr

def learn_J_from_data(datas, N_e=0, signs=None):
    """use linear regression to learn an estimate of the connectivity matrix J"""

    y = np.concatenate(datas, axis = 0)
    y_data = y[:-1,:]
    y_next_data = y[1:,:]

    # J should obey Dale's law so solve for J using constrained linear regression
    if y_data.shape[1] < 100:
        # learn this matrix in a go when # of neurons <=100
        J = cp.Variable((y_data.shape[1],y_data.shape[1]))
        # N_e columns of J should be positive and the rest should be negative
        if signs is None:
            constraints = [J[:,0:N_e] >= 0, J[:,N_e:] <= 0]
        else:
            e_cells = np.where(signs==1)[0]
            i_cells = np.where(signs==2)[0]
            constraints = [J[:,e_cells] >= 0, J[:,i_cells] <= 0]
        # define the objective``
        objective = cp.Minimize(cp.norm(y_next_data.T - J@y_data.T, 'fro'))
        # define the problem
        problem = cp.Problem(objective, constraints)
        # solve the problem
        problem.solve(cp.MOSEK, verbose = False)
        return J.value
    else:
        # learn every row of J separately
        J = np.empty((y_data.shape[1], y_data.shape[1]))
        for i in tqdm(range(y_data.shape[1])):
            J_i = cp.Variable((y_data.shape[1]))
            # N_e columns of J should be positive and the rest should be negative
            if signs is None:
                constraints = [J_i[0:N_e] >= 0, J_i[N_e:] <= 0]
            else:
                e_cells = np.where(signs==1)[0]
                i_cells = np.where(signs==2)[0]
                constraints = [J_i[e_cells] >= 0, J_i[i_cells] <= 0]
            # define the objective
            objective = cp.Minimize(cp.norm(y_next_data.T[i,:] - J_i@y_data.T, 'fro'))
            # define the problem
            problem = cp.Problem(objective, constraints)
            # solve the problem
            problem.solve(cp.MOSEK, verbose = False)
            # put this solution in J
            J[i,:] = J_i.value
        return J
    
def learn_J_from_data_multiregion(datas, signs, region_identity):
    y = np.concatenate(datas, axis = 0)
    y_data = y[:-1,:]
    y_next_data = y[1:,:]

    # make sure we only have two regions
    assert len(np.unique(region_identity)) == 2, "Only two regions are supported"

    # J obeys Dale's law and cross region connectivity pattern
    J = np.empty((y_data.shape[1], y_data.shape[1]))
    for i in tqdm(range(y_data.shape[1])):
        J_i = cp.Variable((y_data.shape[1]))
        # check if this cell is in region 1 or region 2
        if region_identity[i] == 0:
            # this is region 1
            e_cells_this_region = np.where((signs==1) & (region_identity==0))[0]
            i_cells_this_region = np.where((signs==2) & (region_identity==0))[0]
            e_cells_other_region = np.where((signs==1) & (region_identity==1))[0]
            i_cells_other_region = np.where((signs==2) & (region_identity==1))[0]
            if len(e_cells_other_region)>0:
                constraints = [J_i[e_cells_this_region] >= 0, J_i[i_cells_this_region] <= 0, J_i[e_cells_other_region] >= 0, J_i[i_cells_other_region] == 0]
            else:
                constraints = [J_i[e_cells_this_region] >= 0, J_i[i_cells_this_region] <= 0, J_i[i_cells_other_region] == 0]
        else:
            # this is region 2
            e_cells_this_region = np.where((signs==1) & (region_identity==1))[0]
            i_cells_this_region = np.where((signs==2) & (region_identity==1))[0]
            e_cells_other_region = np.where((signs==1) & (region_identity==0))[0]
            i_cells_other_region = np.where((signs==2) & (region_identity==0))[0]
            if len(e_cells_this_region)>0:
                constraints = [J_i[e_cells_this_region] >= 0, J_i[i_cells_this_region] <= 0, J_i[e_cells_other_region] >= 0, J_i[i_cells_other_region] == 0]
            else:
                constraints = [J_i[i_cells_this_region] <= 0, J_i[e_cells_other_region] >= 0, J_i[i_cells_other_region] == 0]
        
        # define the objective
        objective = cp.Minimize(cp.norm(y_next_data.T[i,:] - J_i@y_data.T, 'fro'))
        # define the problem
        problem = cp.Problem(objective, constraints)
        # solve the problem
        problem.solve(cp.MOSEK, verbose = False)
        # put this solution in J
        J[i,:] = J_i.value
    return J

        
def obtain_params_nnmf_init(seed, datas, signs, rank_e, rank_i):
    """ learn initial dynamics and emission matrices by performing structued NNMF on a learned connectivity matrix J"""
    # first learn J from the data
    print('learning connectivity matrix J from data')
    np.random.seed(seed)
    J = learn_J_from_data(datas, signs = signs)
    # now initialize using NNMF
    print('initializing with NNMF')
    N_e = np.sum(signs>0)
    N_i = np.sum(signs<0)
    U, V, error = reconstruct_J_nnmf(seed, J, N_e, N_i, rank_e, rank_i, signs)
    # print error
    print('error during NNMF initialization is '+str(error))
    # we want to set the dynamics matrix to V@U
    A = V@U
    # let's set the emission matrix to U
    C = U
    return A, C

def reconstruct_multiregion_J_nnmf(seed, J, signs, region_identity, rank_e, rank_i):
    ''' perform NNMF on J for region and each cell type separately'''

    assert len(np.unique(signs)) == 2, "Only E and I cell classes are supported"
    # extract rows of J corresponding to region 1 and signs 1
    e_cells_region_1 = np.where((signs==1) & (region_identity==0))[0]
    e_cells_region_2 = np.where((signs==1) & (region_identity==1))[0]
    i_cells_region_1 = np.where((signs==2) & (region_identity==0))[0]
    i_cells_region_2 = np.where((signs==2) & (region_identity==1))[0]

    J_1_e = np.abs(J[e_cells_region_1,:])
    if len(e_cells_region_2)>0:
        J_2_e = np.abs(J[e_cells_region_2,:])
    J_1_i = np.abs(J[i_cells_region_1,:])
    J_2_i = np.abs(J[i_cells_region_2,:])


    # let's run NNMF on each of these matrices
    # let's do multiple runs
    n_runs = 10
    min_error = np.inf
    best_U = None
    best_V = None
    for i in range(n_runs):
        model = NMF(n_components=rank_e, init='random', random_state=seed+i)
        W_1_e = model.fit_transform(J_1_e)
        H_1_e = model.components_
        if len(e_cells_region_2)>0:
            model = NMF(n_components=rank_e, init='random', random_state=seed+i)
            W_2_e = model.fit_transform(J_2_e)
            H_2_e = model.components_
        # let's fit a new model for the inhibitory cells
        model = NMF(n_components=rank_i, init='random', random_state=seed+i)
        W_1_i = model.fit_transform(J_1_i)
        H_1_i = model.components_
        model = NMF(n_components=rank_i, init='random', random_state=seed+i)
        W_2_i = model.fit_transform(J_2_i)
        H_2_i = model.components_

        # now let's reconstruct J
        U = np.zeros((J.shape[0], 2*(rank_e+rank_i)))
        U[e_cells_region_1,:rank_e] = W_1_e
        U[i_cells_region_1,rank_e:rank_e+rank_i] = W_1_i
        if len(e_cells_region_2)>0:
            U[e_cells_region_2,rank_e:2*rank_e] = W_2_e
        # U[e_cells_region_2,rank_e+rank_i:2*rank_e+rank_i] = W_2_e
        U[i_cells_region_2,2*rank_e+rank_i:] = W_2_i

        V = np.zeros((2*(rank_e+rank_i), J.shape[1]))
        V[:rank_e] = H_1_e
        V[rank_e:rank_e+rank_i] = H_1_i
        if len(e_cells_region_2)>0:
            V[rank_e+rank_i:2*rank_e+rank_i] = H_2_e
        V[2*rank_e+rank_i:] = H_2_i

        V[:(rank_e+rank_i),i_cells_region_1] = -V[:(rank_e+rank_i),i_cells_region_1].copy()
        V[(rank_e+rank_i):,i_cells_region_2] = -V[(rank_e+rank_i):,i_cells_region_2].copy()
        J_recovered = U@V

        # now let's compute the error
        error = np.linalg.norm(J_recovered - J, ord='fro')/np.linalg.norm(J, ord='fro')

        if error<min_error:
            min_error = error
            best_U = U
            best_V = V

    return best_U, best_V, min_error



def obtain_params_nnmf_init_multiregion(seed, datas, signs, region_identity, rank_e, rank_i):
    ''' learn initial dynamics and emission matrices by performing structured NNMF on a learned connectivity matrix J'''
    # first learn J from the data
    print('learning connectivity matrix J from data')
    J = learn_J_from_data_multiregion(datas, signs, region_identity)
    # now initialize using NNMF
    print('initializing with NNMF')
    U, V, error = reconstruct_multiregion_J_nnmf(seed, J, signs, region_identity, rank_e, rank_i)
    # print error
    print('error during NNMF initialization is '+str(error))
    # we want to set the dynamics matrix to V@U
    A = V@U
    # let's set the emission matrix to U
    C = U
    return A, C


def shared_rowspace_NNMF(seed, J_e, J_i, rank_e, rank_i, n_iter=10000, tol=1e-3):
    """ Shared / group NNMF, performs NNMF separately on the E and I dyanmics portions of J while keeping the row spaces shared """

    np.random.seed(seed)

    assert rank_e == rank_i, "The ranks of J_e and J_i are not equal"
    rank = rank_e

    # get the dimensions of J_e and J_i
    N_e, N = J_e.shape
    N_i, N = J_i.shape

    # make J_e and J_i positive,
    J_e_positive = np.abs(J_e)
    J_i_positive = np.abs(J_i)

    # initialize the col matrices 
    W_e = np.random.rand(N_e, rank)
    W_i = np.random.rand(N_i, rank)
    # initialize the row matrix
    H = np.random.rand(rank, N)

    # we want to alternately update W_e, W_i, and then H
    for i in range(n_iter):
        # update W_e and W_i
        W_e, W_i = update_W_e_W_i(J_e_positive, J_i_positive, W_e, W_i, H,)
        # update H
        H = update_H(J_e_positive, J_i_positive, W_e, W_i, H,)

        # check for convergence
        if i % 10 == 0:
            # calculate the error
            error = np.linalg.norm(J_e_positive - W_e@H)/np.linalg.norm(J_e_positive) + np.linalg.norm(J_i_positive - W_i@H)/np.linalg.norm(J_i_positive)
            print("Iteration: {}, Error: {}".format(i, error))
            if error < tol:
                print("Converged")
                break

        if i==n_iter-1:
            print("Reached max iterations")

    return W_e, W_i, H


def update_W_e_W_i(J_e, J_i, W_e, W_i, H, lr=0.01):
    # we can hand compute gradients for this
    grad_we = -(J_e - W_e@H)@H.T
    grad_wi = -(J_i - W_i@H)@H.T

    # project the gradients so that W_e and W_i are always positive
    find_inds = np.where(W_e == 0)
    grad_we[find_inds] = np.clip(grad_we[find_inds], a_min=None, a_max=0)
    find_inds = np.where(W_i == 0)
    grad_wi[find_inds] = np.clip(grad_wi[find_inds], a_min=None, a_max=0)

    # # using inverse hessian to set the learning rate
    # lr = 1/(np.diag(H@H.T))[:,None].T
    # lr[np.isnan(lr)] = 0

    W_e = np.maximum( W_e - lr*grad_we, np.zeros_like(W_e))
    W_i = np.maximum(W_i - lr*grad_wi, np.zeros_like(W_i))

    return W_e, W_i


def update_H(J_e, J_i, W_e, W_i, H, lr=0.01):

    # compute gradients
    grad_H = -W_i.T@(J_i - W_i@H) -W_e.T@(J_e - W_e@H)
    find_inds = np.where(H == 0)
    grad_H[find_inds] = np.clip(grad_H[find_inds], a_min=None, a_max=0)

    # # using inverse hessian to set the learning rate
    # lr = 1/(np.diag(W_e.T@W_e + W_i.T@W_i))[:,None]
    # lr[np.isnan(lr)] = 0

    H = np.maximum(H - grad_H*lr, np.zeros_like(H))
    return H

def generate_low_rank_J(seed, N, N_e, N_i, r, diag = False):
    # let's fix the dynamics matrix, assume some low-d structure and set it's effective dimensionality
    # say J = UV
    # let's ensure U has all positive elements
    U = np.random.rand(N,r)
    assert np.linalg.matrix_rank(U)==r, "U doesn't have the appropriate rank"
    # now for J to have N_e positive columns and N_i negative columns, let's try the following
    V_e = np.random.rand(r, N_e)
    # to ensure that V_e and V_i don't have the same elements
    np.random.seed(seed+1)
    V_i = -np.random.rand(r, N_i)
    V = np.hstack((V_e, V_i))
    assert np.linalg.matrix_rank(V)==r, "V doesn't have the appropriate rank"
    # now get J 
    J = U@V 

    if diag:
        # put zeros on the diagonals, however then J is not low-rank anymore
        np.fill_diagonal(J, 0)
     
    # scale J to ensure all eigen values lie within unit circle
    eig_values, _ = np.linalg.eig(J)
    spectral_radius = np.max(np.abs(eig_values))
    J = J/(spectral_radius+0.5)
    return J

def reconstruct_J_nnmf(seed, J, N_e, N_i, rank_e, rank_i, signs=None):
    """ using sklearn's basic NNMF, try to recover the connectivity matrix J"""
   
    # now let's extract the upper half of J 
    if signs is None:
        J_upper = J[:N_e,:]
        J_lower = J[N_e:,:]
    else:
        e_cells = np.where(signs==1)[0]
        i_cells = np.where(signs==2)[0]
        J_upper = J[e_cells,:]
        J_lower = J[i_cells,:]

    # check that there are only two cell classes
    assert len(np.unique(signs)) == 2, "Only E and I cell classes are supported"
        
    # make it positive 
    J_upper_positive = np.abs(J_upper)
    J_lower_positive = np.abs(J_lower)

    # now let's try to recover J_upper_positive and J_lower_positive using NNMF
    # let's do multiple runs
    n_runs = 10
    min_error = np.inf
    best_U = None
    best_V = None
    for i in range(n_runs):
        model = NMF(n_components=rank_e, init='random', random_state=i+seed)
        # let's fit the model to recover J_upper_positive
        W_upper = model.fit_transform(J_upper_positive)
        H_upper = model.components_
        # let's fit the model to recover J_lower_positive
        model = NMF(n_components=rank_i, init='random', random_state=i+seed)
        W_lower = model.fit_transform(J_lower_positive)
        H_lower = model.components_

        # now let's reconstruct J
        if signs is None:
            U = block_diag(W_upper, W_lower)
            V = np.vstack((H_upper, H_lower))
            # make the right half of V negative
            V[:,N_e:] = -V[:,N_e:].copy()
        else:
            U = np.zeros((N_e+N_i, rank_e+rank_i))
            U[e_cells,:rank_e] = W_upper
            U[i_cells,rank_e:] = W_lower
            V = np.zeros((rank_e+rank_i, N_e+N_i))
            V[:rank_e] = H_upper
            V[rank_e:] = H_lower
            V[:,i_cells] = -V[:,i_cells].copy()
        J_recovered = U@V

        # now let's compute the error
        error = np.linalg.norm(J_recovered - J, ord='fro')/np.linalg.norm(J, ord='fro')

        if error<min_error:
            min_error = error
            best_U = U
            best_V = V

    return best_U, best_V, min_error


def reconstruct_shared_nnmf_J(J, N_e, N_i, rank_e, rank_i):
    """ perform NNMF on the top and botom halves of J seperately while constraining their row matrices to be shared """

    # now let's extract the upper half of J 
    J_upper = J[:N_e,:]
    # make it positive 
    J_upper_positive = np.abs(J_upper)


    # now let's extract the lower half of J
    J_lower = J[N_e:,:]
    # make it positive
    J_lower_positive = np.abs(J_lower)

    # now let's try to recover J_upper_positive and J_lower_positive using NNMF
    # let's do multiple runs
    n_runs = 10
    min_error = np.inf
    best_U = None
    best_V = None
    for i in range(n_runs):
        W_upper, W_lower, H = shared_rowspace_NNMF(i, J_upper_positive, J_lower_positive, rank_e, rank_i)

        if W_upper is None or W_lower is None or H is None:
            continue

        # now let's reconstruct J
        U = block_diag(W_upper, W_lower)
        V = np.vstack((H, H))
        # make the right half of V negative
        V[:,N_e:] = -V[:,N_e:].copy()
        J_recovered = U@V
       
        # now let's compute the error
        error = np.linalg.norm(J_recovered - J, ord='fro')/np.linalg.norm(J, ord='fro')

        if error<min_error: 
            min_error = error
            best_U = U
            best_V = V

    return best_U, best_V, min_error


def initialize_with_nnmf(model, init_nnmf, datas, inputs=None, masks=None, tags=None,):
    """ Initialize the model parameters using NNMF """
     # First initialize the observation model, with the first element in the NNMF list

    emission_init = init_nnmf[1].T
    # Assign each state a random projection of the emission_init
    Keff = 1 if model.emissions.single_subspace else model.K
    Cs, ds = [], []
    for k in range(Keff):
        weights = npr.randn(model.D, model.D * Keff)
        weights = np.linalg.svd(weights, full_matrices=False)[2]
        if Keff == 1:
            Cs.append(init_nnmf[1])
        else:
            Cs.append((weights @ emission_init).T)
        ds.append(np.mean(np.concatenate(datas), axis=0))

    model.emissions.Cs = np.array(Cs)
    model.emissions.ds = np.array(ds)

    # Get the initialized variational mean for the data
    xs = [model.emissions.invert(data, input, mask, tag)
            for data, input, mask, tag in zip(datas, inputs, masks, tags)]
    xmasks = [np.ones_like(x, dtype=bool) for x in xs]
    
    # no compute the noise in y = Cx + d+ noise
    noise = np.concatenate([data - x@C.T - d for data, x, C, d in zip(datas, xs, Cs, ds)])
    # find the variance of each dimension
    var = np.var(noise, axis=0)
    # set the variance of the emission model
    model.emissions.inv_etas[:,...] = np.diag(var) + 1e-5*np.eye(model.N)

    # now initialize the dynamics model
    # only valid for k=1
    assert model.K==1, "Currently only valid for K=1"

    model.dynamics.As = np.array([init_nnmf[0]])
    # compute the bias as the mean of the xs
    model.dynamics.bs = np.array([np.mean(np.concatenate(xs), axis=0)])
    # now let's compute noise
    noise = np.concatenate([x[1:] - x[:-1]@A.T - b for x, A, b in zip(xs, model.dynamics.As, model.dynamics.bs)])
    # find the variance of each dimension
    var = np.var(noise, axis=0)
    # set the variance of the dynamics model
    # model.dynamics._sqrt_Sigmas = np.array([np.diag(np.sqrt(var))])
    model.dynamics._sqrt_Sigmas = np.array([np.eye(model.D) + 1e-5*np.eye(model.D)])



if __name__ == "__main__":
    # check if running on cluster
    CLUSTER = False
    PLOT = True
    SAVE = False

    # take arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--nnmf_type', type=str, default='standard', help='standard or shared')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--r', type=int, default=4)

    args = parser.parse_args()

    # take seed as an argument from command line
    if CLUSTER:
        seed = int(os.environ['SLURM_ARRAY_TASK_ID'])
    else:
        seed = 2
    # fix a seed
    np.random.seed(seed)

    # we want to simulate N neurons with the dynamics: y = Jy + noise
    N = args.N
    # let's set the number of e and i cells
    N_e  = int(N/2)
    N_i  = N - N_e

    # create a low rank connectivity matrix
    # first choose the effective
    #  dimensionality of the dynamics
    r = args.r
    J = generate_low_rank_J(seed, N, N_e, N_i, r)

    # now let's try to recover J using NNMF
    if args.nnmf_type == 'standard':
        U, V, error = reconstruct_J_nnmf(J, N_e, N_i)
    elif args.nnmf_type == 'shared':
        U, V, error = reconstruct_shared_nnmf_J(J, N_e, N_i)

    J_rec = U@V
    # print the error
    print("error: ", error)

    if SAVE:
        # save this error
        np.save("results_lds_constraints/"+str(args.nnmf_type) + "_nnmf_error_J_"+str(args.N)+"_seed_"+str(seed)+".npy", error)

    if PLOT:
        # also plot the eigen values of computed and recovered J
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(np.real(np.linalg.eigvals(J)), np.imag(np.linalg.eigvals(J)), label='J', alpha=0.5)
        # set x and y limits between -1 and 1
        ax[0].set_xlim(-1,1)
        ax[0].set_ylim(-1,1)
        # set title to J
        ax[0].set_title("J")

        ax[1].scatter(np.real(np.linalg.eigvals(J_rec)), np.imag(np.linalg.eigvals(J_rec)), label='J_rec', alpha=0.5)
        # set x and y limits between -1 and 1
        ax[1].set_xlim(-1,1)
        ax[1].set_ylim(-1,1)
        # set title to J_rec
        ax[1].set_title("J_rec")
        plt.tight_layout()
        fig.suptitle("eigen values of J and J_rec")

        plt.show()

        # plot J and J_rec side by side
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(J, cmap='bwr')
        ax[0].set_title("J")
        ax[1].imshow(J_rec, cmap='bwr')
        ax[1].set_title("J_rec")
        plt.tight_layout()
        fig.suptitle("J and J_rec")
        plt.show()


