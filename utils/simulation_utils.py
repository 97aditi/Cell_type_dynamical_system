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


def create_dynamics_matrix(list_of_dimensions):
    """ 
    Creates a multi-region dynamics matrix compliant with Dale's law, and only excitatory cross-region connections.
    
    Parameters:
    list_of_dimensions (numpy array): of size num_regions x 2, where the first column is 
                                       the number of excitatory latents for the region 
                                       and the second column is the number of inhibitory latents 
                                       for the region.
    
    Returns:
    numpy.ndarray: The dynamics matrix for the network.
    """
    
    num_regions = len(list_of_dimensions)
    assert num_regions >= 1, "At least 1 region is required"
    
    # Initialize the size of the dynamics matrix
    total_latents = np.sum(list_of_dimensions, axis=0)[0] + np.sum(list_of_dimensions, axis=0)[1]
    A = np.zeros((total_latents, total_latents))
    
    current_index = 0
    
    # Create A_ii blocks (within-region dynamics)
    for i in range(num_regions):
        excitatory_latents, inhibitory_latents = list_of_dimensions[i]
        num_latent_per_region = excitatory_latents + inhibitory_latents

        # Create the within-region dynamics matrix (A_ii)
        A_ii = np.zeros((num_latent_per_region, num_latent_per_region))
        
        # Fill excitatory connections
        A_ii[:, :excitatory_latents] = np.random.rand(num_latent_per_region, excitatory_latents)
        # Fill inhibitory connections
        A_ii[:, excitatory_latents:num_latent_per_region] = -np.random.rand(num_latent_per_region, inhibitory_latents)
        
        # Add positive biases for stabilit
        A_ii = 0.5 * np.identity(num_latent_per_region) + 0.5 * A_ii
        
        # Normalize A_ii and check for NaNs or Infs
        max_eigval = np.max(np.abs(np.linalg.eigvals(A_ii)))
        if max_eigval != 0:
            A_ii /= max_eigval  # Normalize for stability

        # Place A_ii in the correct block location
        A[current_index:current_index + num_latent_per_region,
          current_index:current_index + num_latent_per_region] = A_ii
        
        current_index += num_latent_per_region


    current_index_i = 0
    # Create A_ij blocks (between-region dynamics)
    for i in range(num_regions):
        excitatory_latents_i, _ = list_of_dimensions[i]
        num_latent_per_region_i = np.sum(list_of_dimensions[i])
        current_index_j = 0
        for j in range(num_regions):
            excitatory_latents_j, _ = list_of_dimensions[j]
            num_latent_per_region_j = np.sum(list_of_dimensions[j])
            if i != j:
                # Initialize the between-region dynamics with zeros
                A_ij = np.zeros((num_latent_per_region_i, num_latent_per_region_j))
                
                # Fill only the excitatory to excitatory connections
                A_ij[:, :excitatory_latents_j] = np.random.rand(num_latent_per_region_i, excitatory_latents_j)

                # The connections along the inhibitory dimensions will remain zero, which is already the case
                
                # Normalize to reduce connectivity strength
                max_val = np.max(A_ij)
                if max_val != 0:
                    A_ij /= (10 * max_val)  # Scale down excitatory connections
                
                # Place A_ij in the correct location
                A[current_index_i :current_index_i + num_latent_per_region_i,
                  current_index_j :current_index_j + num_latent_per_region_j] = A_ij
            current_index_j += num_latent_per_region_j
        
        current_index_i += num_latent_per_region_i

    # Normalize the entire dynamics matrix
    max_eigval = np.max(np.abs(np.linalg.eigvals(A)))
    if max_eigval != 0:
        A /= max_eigval  # Normalize to keep it stable

    return A