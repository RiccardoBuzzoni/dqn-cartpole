import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    
    This neural network takes the environment's state as input and outputs a Q-value for each possible action.
    The action that the agent will choose to perform is the one with the highest Q-value.
    
    Architecture: Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Initialization of DQN layers.
        
        Args:
            state_size (int): Number of values that describe the environment state.
                For CartPole, there are 4: [cart position, cart velocity, pole angle, pole angle velocity]
                    
            action_size (int): Number of values that describe actions the agent can perform.
                For CartPole, there are 2: [push left, push right]
                
            hidden_size (int): Number of neurons in the hidden layers, 128 by default.
        """
        super(DQN, self).__init__()

        # --- Build the neural network as a sequential stack of layers ---
        self.model = nn.Sequential(

            # --- Input layer ---
            # Takes the raw state and expands it to 'hidden_state' neurons so the network can learn complex patterns.
            nn.Linear(state_size, hidden_size),

            # Activation function: ReLU sets all negative values to 0.
            # This introduces non-linearity, allowing the network to learn more than simple straight-line relationships.
            nn.ReLU(),

            # --- Hidden layer ---
            # Processes the features learn by the Input Layer firther. Both input and output are hidden_size neurons.
            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            # --- Output Layer ---
            # Compresses everything down to a single value per action. Each output neuron represents the Q-value of one action.
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes the Q-value for a given state.
        
        This method is called automatically when model(state) is called. It passes the state through all the layers defined above.
        
        Args:
            state (torch.Tensor): A tensor which represents the current environment state.
            
        Returns:
            torch.Tensor: A tensor of Q-values, one for each action.
                        e.g tensor([-0.32, 1.57 ]):
                        Q(left) = -0.32, Q(right) = 1.57 -> The agents picks "right.
        """
        return self.model(state)
