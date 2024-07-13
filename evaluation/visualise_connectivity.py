import numpy as np
import matplotlib.pyplot as plt


def reorder_connectivity_matrix(connectivity_data, connectivity_covariance_matrix, rank):
    input_vector = connectivity_data[:, int(2*rank):-1]
    input_selection_vector = connectivity_data[:, 0:rank]
    output_vector = connectivity_data[:, rank:int(2*rank)]
    readout_vector = connectivity_data[:, -1, np.newaxis]

    # re-order data for plotting 
    connectivity_data_xaxis = \
        np.hstack((input_vector, input_selection_vector, output_vector))
    connectivity_data_yaxis = \
        np.hstack((input_selection_vector, output_vector, readout_vector))
    connectivity_covariance_matrix_reordered_tent = \
        np.delete(connectivity_covariance_matrix, [-2], axis=1)    
    connectivity_covariance_matrix_reordered_tent= \
        np.delete(connectivity_covariance_matrix_reordered_tent, [-1], axis=0)    
    connectivity_covariance_matrix_reordered = \
        np.copy(connectivity_covariance_matrix_reordered_tent)
    connectivity_covariance_matrix_reordered =\
        np.vstack((connectivity_covariance_matrix_reordered[-1,:], 
                connectivity_covariance_matrix_reordered[0:-1, :]))

    return connectivity_data_xaxis, connectivity_data_yaxis, connectivity_covariance_matrix_reordered

def create_list_vector_names(rank):

    vector_names_xaxis = ['I']
    vector_names_xaxis.extend([rf'n$^{i}$' for i in range(1, rank+ 1)])
    vector_names_xaxis.extend([rf'm$^{i}$' for i in range(1, rank+ 1)])

    vector_names_yaxis = [rf'n$^{i}$' for i in range(1, rank+ 1)]
    vector_names_yaxis.extend([rf'm$^{i}$' for i in range(1, rank+ 1)])
    vector_names_yaxis.extend(['W'])

    return vector_names_xaxis, vector_names_yaxis


def plot_neurons_connectivity_space(connectivity_data, connectivity_covariance_matrix, rank):
    connectivity_data_xaxis, connectivity_data_yaxis, connectivity_covariance_matrix_reordered = \
        reorder_connectivity_matrix(connectivity_data, connectivity_covariance_matrix, rank)

    vector_names_xaxis, vector_names_yaxis = create_list_vector_names(rank)

    fig, axes = \
        plt.subplots(connectivity_covariance_matrix_reordered.shape[1], 
                    connectivity_covariance_matrix_reordered.shape[1], figsize=(8,8))
    for i_subplot in range(connectivity_covariance_matrix_reordered.shape[1]):
        for j_subplot in range(connectivity_covariance_matrix_reordered.shape[1]):
            if i_subplot <= j_subplot:
                axes[i_subplot, j_subplot].scatter(connectivity_data_xaxis[:, i_subplot], 
                                                   connectivity_data_yaxis[:, j_subplot])
                axes[i_subplot, j_subplot].set_xlabel('{} [-]'.format(vector_names_xaxis[i_subplot]))
                axes[i_subplot, j_subplot].set_ylabel('{} [-]'.format(vector_names_yaxis[j_subplot]))
            else:
                fig.delaxes(axes[i_subplot, j_subplot])

    plt.tight_layout()
    plt.show()


def plot_connectivity_covariance_matrix(connectivity_data, connectivity_covariance_matrix, rank):
    
    _, _, connectivity_covariance_matrix_reordered = \
        reorder_connectivity_matrix(connectivity_data, connectivity_covariance_matrix, rank)

    vector_names_xaxis, vector_names_yaxis = create_list_vector_names(rank)

    # show covariance of connectivity vectors
    mask =  np.tri(connectivity_covariance_matrix_reordered.shape[0], k=-1)
    connectivity_covariance_matrix_reordered= \
        np.ma.array(connectivity_covariance_matrix_reordered, mask=mask)
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad('white')
    fig, ax = plt.subplots()
    im = ax.imshow(connectivity_covariance_matrix_reordered, aspect='auto', cmap=cmap, clim=[-2.2, 2.2])
    ax.set_xticks(range(len(vector_names_yaxis)), labels=vector_names_yaxis)
    ax.set_yticks(range(len(vector_names_xaxis)), labels=vector_names_xaxis)
    fig.colorbar(im, ax=ax, label='covariance [-]')
    ax.yaxis.tick_right()
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.show()


def plot_resampled_neurons_connectivity_space(resampled_connectivity_data, connectivity_data, connectivity_covariance_matrix, rank):
    
    connectivity_data_xaxis, connectivity_data_yaxis, connectivity_covariance_matrix_reordered = \
        reorder_connectivity_matrix(connectivity_data, connectivity_covariance_matrix, rank)

    vector_names_xaxis, vector_names_yaxis = create_list_vector_names(rank)

    resampled_connectivity_data_xaxis = \
        np.hstack((resampled_connectivity_data[:, -2, np.newaxis], resampled_connectivity_data[:, :-2]))
    resampled_connectivity_data_yaxis = \
        np.hstack((resampled_connectivity_data[:, :-2], resampled_connectivity_data[:, -1, np.newaxis]))

    fig, axes = \
        plt.subplots(connectivity_covariance_matrix_reordered.shape[1], 
                    connectivity_covariance_matrix_reordered.shape[1], figsize=(8,8))
    for i_subplot in range(connectivity_covariance_matrix_reordered.shape[1]):
        for j_subplot in range(connectivity_covariance_matrix_reordered.shape[1]):
            if i_subplot <= j_subplot:
                axes[i_subplot, j_subplot].scatter(connectivity_data_xaxis[:, i_subplot], 
                                                connectivity_data_yaxis[:, j_subplot], 
                                                label='original neurons')
                axes[i_subplot, j_subplot].scatter(resampled_connectivity_data_xaxis[:, i_subplot], 
                                                resampled_connectivity_data_yaxis[:, j_subplot], 
                                                label='resmapled neurons')
                axes[i_subplot, j_subplot].set_xlabel('{} [-]'.format(vector_names_xaxis[i_subplot]))
                axes[i_subplot, j_subplot].set_ylabel('{} [-]'.format(vector_names_yaxis[j_subplot]))
            else:
                fig.delaxes(axes[i_subplot, j_subplot])

    plt.tight_layout()
    plt.legend(loc='center', bbox_to_anchor=(-1, 0.5))
    plt.show()