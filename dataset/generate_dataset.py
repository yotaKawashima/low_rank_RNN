import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class PDMStimulus(Dataset):
    def __init__(self, num_trials, t_onset, t_offset, t_max, u_mean_list, sampling_rate, noise_std = 0.03):
        """
        Arguments:
            num_trials (int): the number of trials
            t_onset (float): stimulus ramp-up onset [s] 
            t_offset (float): stimulus ramp-up offset [s]
            t_max (float): max time [s]
            u_mean_list (np.ndarray):  a list of mean stimulus intensity 
            sampling_rate (float): sampling rate [Hz]
        Return
            u (np.ndarray): the stimulus intensity for each trial
        """
        #super(PDMStimulus, self).__init__()
        self.num_trials = num_trials
        self.t_onset = t_onset
        self.t_offset = t_offset
        self.t_max = t_max
        self.u_mean_list = u_mean_list
        self.sampling_rate = sampling_rate # Hz
        self.num_sample_points = int(self.t_max * self.sampling_rate)
        self.time = np.linspace(0, self.t_max, self.num_sample_points)
        self.stimuli, self.labels = self.generate_data(noise_std)        

    def generate_data(self, noise_std=0.03):
        """
        Generate stimulus for each trial
        """
        # mean stimulus intensity for each trial 
        u_mean = np.random.choice(self.u_mean_list, self.num_trials)

        # noise for each trial N(0, 0.03)
        noise = \
            np.random.normal(0, noise_std, (self.num_trials, self.num_sample_points))
        
        # initialise the stimulus intensity
        u = np.empty((self.num_trials, self.num_sample_points))  

        # add noise to the stimulus intensity  
        u[:] = noise
        
        # convert sec to sample point locations
        t_onset_sample = int(self.t_onset * self.sampling_rate)
        t_offset_sample = int(self.t_offset * self.sampling_rate) 
        
        # if 5 <= time <=45, then add the mean stimulus intensity
        if self.num_trials == 1:
            u[0, t_onset_sample:t_offset_sample+1] += np.repeat(u_mean, (t_offset_sample-t_onset_sample+1))
            print(u.shape)
        else:
            u[:, t_onset_sample:t_offset_sample+1] += np.tile(u_mean, (t_offset_sample-t_onset_sample+1, 1)).T

        # obtain label for each trial 
        labels = np.sign(u_mean) 

        return u, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        stimulus = self.stimuli[idx, :]
        label = self.labels[idx]
        sample = {'stimulus': stimulus, 'label': label}

        return sample
    
    def plot_stimulus(self, trials=None):
        if trials is None:
            trials = range(self.num_trials)

        fig = plt.figure()
        
        if isinstance(trials, int):
            plt.plot(self.time, self.stimuli[trials, :], \
                     label='trial {:d}, label {:d}'.format(trials, int(self.labels[trials])))
        else:
            for i_trial in trials:
                plt.plot(self.time, self.stimuli[i_trial, :], \
                        label='trial {:d}, label {:d}'.format(int(i_trial), int(self.labels[i_trial])))
        plt.xlim(0, self.t_max)
        plt.xlabel('time [s]')
        plt.ylabel('u')
        plt.legend()
        return fig