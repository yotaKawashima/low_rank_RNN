import numpy as np
import scipy 

class LatentDynamics:
    def __init__(self, rank, connectivity_covariance_matrix, time_step_size):
        """
        rank (scaler): rank 
        connectivity_covariance_matrix (ndarray): covarianc matrix of connectivity
        time_step_size (scaler): dt  
        """
        self.rank = rank 
        self.connectivity_covariance_matrix = connectivity_covariance_matrix
        self.time_step_size = time_step_size 
        # keep tentative tilda_sigma to compute tilda_sigma in euler method
        self.tilda_sigma_mn_tentative = np.zeros((rank, rank))
        self.tilda_sigma_nI_tentative = np.zeros((rank, rank))
        for i_rank in range(rank):
            self.tilda_sigma_mn_tentative[i_rank, i_rank] = \
                np.sqrt(self.connectivity_covariance_matrix[i_rank, i_rank+rank])
            self.tilda_sigma_nI_tentative[i_rank, i_rank] = \
                np.sqrt(self.connectivity_covariance_matrix[i_rank, -2])
            
        # Extract variance
        self.var_ms = np.diagonal(self.connectivity_covariance_matrix[rank:-2, rank:-2])
        self.var_I = self.connectivity_covariance_matrix[-2, -2]

    def run_simulation(self, stimulus_data, label_data):
        """
        stimulus_data (ndarray): trial x time
        Label_data (ndarray): trial x time
        """
        num_sample_points = stimulus_data.shape[1]
        num_trials = stimulus_data.shape[0]

        # initialise array to keep data (rank x times)
        ks_history = \
            np.zeros((num_trials, self.rank, num_sample_points))
        tilda_sigma_mns_history = \
            np.zeros((num_trials, self.rank, num_sample_points))
        tilda_sigma_nIs_history = \
            np.zeros((num_trials, self.rank, num_sample_points))
        averaged_neural_gain_history = \
            np.zeros((num_trials, num_sample_points))
        
        for i_trial in range(num_trials):
            stimulus_data_this_trial = stimulus_data[i_trial, :] 
            label_data_this_trial = label_data[i_trial]
            
            # initial ks 
            ks = ks_history[i_trial, :, 0]

            # Run euler method 
            for i_time in range(num_sample_points):
                input_data = stimulus_data_this_trial[i_time]
                ks, tilda_sigma_mn, tilda_sigma_nI, averaged_neural_gain = \
                    self.euler_method(ks, input_data)
                # store data
                ks_history[i_trial, :, i_time] = ks
                tilda_sigma_mns_history[i_trial, :, i_time] = np.diagonal(tilda_sigma_mn) 
                tilda_sigma_nIs_history[i_trial, :, i_time] = np.diagonal(tilda_sigma_nI)
                averaged_neural_gain_history[i_trial, i_time] = averaged_neural_gain
        
        self.ks_history = ks_history 
        self.tilda_sigma_mns_history = tilda_sigma_mns_history
        self.tilda_sigma_nIs_history = tilda_sigma_nIs_history
        self.averaged_neural_gain_history = averaged_neural_gain_history


    def euler_method(self, ks, input_data):
        """ Apply euler method to latent dynamics
        ks (ndaary): latent variables at a given time (rank, 1)
        input_data (scaler): stimulus at a given time from one trial
        """
        # compute derivative
        derivative_ks, tilda_sigma_mn, tilda_sigma_nI, averaged_neural_gain = \
            self._derivative_ks(ks, input_data, store_all=True)
        
        # euler method to compute ks
        ks = ks + self.time_step_size * derivative_ks
        
        return ks,  tilda_sigma_mn, tilda_sigma_nI, averaged_neural_gain

    def _derivative_ks(self, ks, input_data, store_all=False):
        averaged_neural_gain = self.average_neural_gain(ks, input_data)
        tilda_sigma_mn = self.tilda_sigma_mn_tentative * averaged_neural_gain
        tilda_sigma_nI = self.tilda_sigma_mn_tentative * averaged_neural_gain
        
        derivative_ks = (- ks + tilda_sigma_mn @ ks + tilda_sigma_nI @ np.ones(self.rank) * input_data)
        
        if store_all:
            return derivative_ks, tilda_sigma_mn, tilda_sigma_nI, averaged_neural_gain
        else:
            return derivative_ks

    def phi_prime(self, x):
        """ Neural gain """
        return 1 - np.tanh(x)**2

    def integrand_for_average_neural_gain(self, z, ks, v):
        """ 
        This function is used for computing the average neural gain.
        z (scalar): random variable to explain connectivity vector,
        such as m, n, I, and W. 
        ks (ndarray): latent variables. (rank, 1)
        v (scalar): input. Assume it is a single input for now.
        """
        delta = np.sqrt(np.dot(self.var_ms, np.square(ks)) + self.var_I*np.square(v))
        neural_gain = self.phi_prime(z * delta)
        return np.exp(-np.square(z)/2) * neural_gain

    def average_neural_gain(self, ks, v):
        """ 
        This function copmputes the average neural gain.
        ks (ndarray): latent variables. (rank, 1)
        v (scalar): input. Assume it is a single input for now.
        """
        average_data, _ = \
            scipy.integrate.quad(self.integrand_for_average_neural_gain, 
                                -np.inf, np.inf, 
                                args=(ks, v,)) / np.sqrt(2*np.pi)
        return average_data


    def energy_function(self, ks, input_data):
        """ Compute energy function
        ks (ndaary): latent variables at a given time (rank, 1)
        input_data (scaler): stimulus at a given time from one trial
        """
        derivative_ks = self._derivative_ks(ks, input_data, store_all=False)
        
        energy = np.sqrt(np.dot(derivative_ks, derivative_ks))/2
        
        return energy.squeeze()

    def minimise_energy(self, initial_ks, input_data):
        ks_optimised = scipy.optimize.minimize(self.energy_function, initial_ks, args=(input_data,))
        return ks_optimised