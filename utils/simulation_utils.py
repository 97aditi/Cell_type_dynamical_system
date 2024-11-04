import numpy as np

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