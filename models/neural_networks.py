import torch
from torch import nn 

# Linear layer
class LinearAmpFactor(nn.Module):
    def __init__(self, input_size, output_size, init_std):
        super(LinearAmpFactor, self).__init__()
        # 
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.init_std = init_std
        
        # weight natrix
        self.weight = \
            torch.normal(0, init_std, (output_size, input_size), dtype=torch.float32)
        #self.amplitude_factor = \ 
        #   nn.Parameter(torch.normal(0, 1, (), dtype=torch.float32)) 
       
    def forward(self, input): 
        output = input @ self.weight.T
        return output.squeeze().float()
    
    def extra_repr(self) -> str:
        return f"in_features={self.input_size}, out_features={self.output_size}"
    
# RNN layer
class RNNLayer(nn.Module):
    def __init__(self, hidden_size, rank, time_step_size, tau):
        super(RNNLayer, self).__init__()
        # 
        # parameters
        self.hidden_size = hidden_size
        self.rank = rank
        self.time_step_size = time_step_size
        self.tau = tau

        # weight natrix
        self.left_singular_vector = \
            nn.Parameter(torch.normal(0, 1, (hidden_size, rank), dtype=torch.float32))
        self.right_singular_vector = \
            nn.Parameter(torch.normal(0, 1, (rank, hidden_size), dtype=torch.float32))      

    def forward(self, feedforward_input, hidden_state):
        
        # compute weights
        self.connectivity_matrix = \
            torch.mm(self.left_singular_vector, self.right_singular_vector) / self.hidden_size  

        # recurrent (euler method for hidden state)
        derivative_hidden = \
            (- hidden_state + \
            (torch.tanh(hidden_state) @ self.connectivity_matrix.T) + \
            feedforward_input.unsqueeze(dim=0)) / self.tau 
        ##print('hidden_state')
        #print(hidden_state)
        #print('tanh(hidden_state)')
        #print(torch.tanh(hidden_state))
        #print('feedforward_input')
        #print(feedforward_input)
                
        #print('derivative_hidden')
        #print(derivative_hidden)
        hidden_state = \
            hidden_state + self.time_step_size * derivative_hidden
        
        return hidden_state.squeeze().float()
    
    def extra_repr(self) -> str:
        return f"in_features={self.hidden_size}, out_features={self.hidden_size}"


# Recurrent Neural Network 
class LowRankRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rank, time_step_size, tau):
        super(LowRankRNN, self).__init__()
        
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rank = rank
        self.time_step_size = time_step_size
        self.tau = tau

        # Initialise the parameters
        #self.input_vector = \
        #    torch.normal(0, 1, (hidden_size, input_size), dtype=torch.float32)
        #self.readout_vector = \
        #    torch.normal(0, 4, (output_size, hidden_size), dtype=torch.float32)

        # network structure
        #self.feedforward_input = nn.Linear(input_size, hidden_size, bias=False)
        #self.feedforward_input.weight = self.input_vector
        self.feedforward_input = LinearAmpFactor(input_size, hidden_size, 1)
        self.recurrent_input = RNNLayer(hidden_size, rank, time_step_size, tau)
        #self.readout = nn.Linear(hidden_size, output_size, bias=False)
        #self.readout.weight = self.readout_vector
        self.readout = LinearAmpFactor(hidden_size, output_size, 4)

    def forward(self, input, hidden_state):
        # feedforward 
        feedforward_input = self.feedforward_input(input)
        #print(feedforward_input)
        
        #print('shape of hidden_state')
        #print(hidden_state.shape)
        #print(hidden_state)
        # recurrent (euler method for hidden state)        
        hidden_state = self.recurrent_input(feedforward_input, hidden_state)
        
        # readout 
        #print('shape of hidden_state')
        #print(hidden_state.shape)
        #print(hidden_state.data)
        #print(torch.tanh(hidden_state.data))

        output = self.readout(torch.tanh(hidden_state)) / self.hidden_size
        return output.float(), hidden_state 
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)
    