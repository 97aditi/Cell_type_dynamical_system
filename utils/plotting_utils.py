import matplotlib.pyplot as plt
import numpy as np
import matplotlib

def plot_and_save_parameters(A, C, Q, R, save_folder, J=None):
    """ plot and save the parameters of the simulation """
    mc = matplotlib.colors
    # plot and save all the matrices in one figure
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    vmin = np.min(A)
    vmax = np.max(A)
    color_norm = mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin = vmin)
    plt.imshow(A, norm=color_norm, cmap="bwr")
    # plt.imshow(A, cmap='bwr', aspect='auto', vcenter=0)
    plt.xticks([])
    plt.yticks([])
    plt.title('A', fontsize=20)
    cbar = plt.colorbar()
    # cbar.set_ticks([-1.0,0,1.0])
    # cbar.set_ticklabels(['-1.0','0','1.0'])
    plt.subplot(2, 2, 2)
    vmax = np.max(C)
    color_norm = mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin = -1)
    plt.imshow(C, cmap='bwr', norm=color_norm)
    plt.title('C', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    # cbar.set_ticks([-1.0,0,1.0])
    # cbar.set_ticklabels(['-1.0','0','1.0'])
    plt.subplot(2, 2, 3)
    plt.imshow(Q, cmap='bwr')
    plt.title('Q', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.subplot(2, 2, 4)
    plt.imshow(R, cmap='bwr')
    plt.title('R', fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()

    if save_folder is not None:
        plt.savefig(save_folder+'model_params.png', bbox_inches='tight', dpi=300)
    if J is not None:
        plt.figure(figsize=(6, 4))
        vmin = np.min(J)
        vmax = np.max(J)
        color_norm = mc.TwoSlopeNorm(vmax=vmax, vcenter=0, vmin = vmin)
        plt.imshow(J, cmap='bwr', norm=color_norm)
        plt.title('J', fontsize=20)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.tight_layout()
        if save_folder is not None:
            plt.savefig(save_folder[:-4] + '_J.png', bbox_inches='tight', dpi=300)
        plt.show()