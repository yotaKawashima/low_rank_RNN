import torch 
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamics_each_trial_PDM(device, model, t_max, binary_dataset, dataloader, num_trial_to_plot=5):
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch == 0:
                _batch_size = len(sample_batched['stimulus'])
                hidden_state = model.init_hidden(_batch_size)
                num_sample_points = sample_batched['stimulus'].shape[1]

                output_list = np.empty((_batch_size, num_sample_points, model.output_size))
                hidden_list = np.empty((_batch_size, num_sample_points, model.hidden_size))
                
                for i_time in range(num_sample_points):
                    input_data = sample_batched['stimulus'][:, i_time].unsqueeze(dim=1).to(device)
                    label_data = sample_batched['label'].float().to(device)
                    hidden_state = hidden_state.to(device)
                    output, hidden_state = model(input_data, hidden_state)

                    hidden_list[:, i_time, :] = hidden_state.cpu().detach()
                    output_list[:, i_time, :] = output.unsqueeze(dim=1).cpu().detach()
                    
                for i_trial in range(num_trial_to_plot):
                    plt.subplot(211)
                    plt.title('trial {} in batch {}'.format(i_trial, i_batch))
                    plt.plot(binary_dataset.time, sample_batched['stimulus'][i_trial, :], color='tab:orange', label='input')
                    plt.plot(binary_dataset.time, output_list[i_trial, :, 0], color='tab:blue', label='output')
                    plt.hlines(label_data[i_trial].cpu().detach(), 0, max(binary_dataset.time), color='tab:green', label='label')
                    plt.legend()
                    plt.xlim(0, t_max)
                    plt.ylabel('[-]')
                    plt.subplot(212)
                    plt.plot(binary_dataset.time, hidden_list[i_trial, :, :])
                    plt.xlabel('time [s]')
                    plt.ylabel('hidden state [-]')
                    plt.xlim(0, t_max)
                    plt.show()
                break