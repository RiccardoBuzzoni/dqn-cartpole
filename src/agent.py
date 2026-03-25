import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    """
    Deep Q-Network Agent.
    
    The agent is the decision-maker: it observes the environment state, chooses an action, stores the experience, and
    learns from past experiences by updating the Q-Network weights.
    
    It uses two key techniques from the original DQN paper (DeepMind, 2015):
        1. Experience Replay: learning from random past experiences (via ReplayBuffer).
        2. Target Network:  a separate, stable copy of the network used to compute learning targets, preventing 
                            oscillations during training.
    """
    def __init__(
            self,
            state_size: int,
            action_size: int,
            hidden_size: int = 128,
            lr: float = 1e-3,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            epsilon_min: float = 0.01,
            epsilon_decay: float = 0.995,
            buffer_capacity: int = 10_000,
            batch_size: int = 64,
            target_update_freq: int = 10,
    ):
        """
        Initialise the DQN Agent.
        
        Args:
            state_size (int): Number of environment states (4 states for CartPole).
            action_size (int): Number of possibile actions (2 actions for CartPole).
            hidden_size (int): Neurons in hidden layers of the network.
            lr (float): Learning rate for optimiser (Controls how big each weight update step is).
            gamma (float): Discount factor for future rewards.
                0 = Agent only focuses on immediate rewards.
                1 = Agent values future rewards equally to immediate ones.
                0.99 = Slight preference for sooner rewards.
            epsilon (float): Starting exploration rate (1.0 for 100% random actions).
            epsilon_min (float): Minimum exploration rate.
                The agent always keeps a small chance of exploring even when well trained.
            epsilon_decay (float): Multiplicative decay applied to epsilon after each episode.
            buffer_capacity (int): Max number of experiences stored in the Replay Buffer.
            batch_size (int): Number of experiences sampled per training step.
            target_update_freq (int): How many episodes between each target network update.
        """
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # --- Epsilon-greedy exploration parameters ---
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # --- Target network frequency update ---
        self.target_update_freq = target_update_freq

        # Automatically select best available device:
        #   - "cuda" if an NVIDIA GPU is available (faster training)
        #   - "cpu" otherwise (perfectly fine for CartPole)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on: {self.device}")

        # --- Networks ---
        # The 'online' network is trained at every step and used to select actions.
        self.online_net = DQN(state_size, action_size, hidden_size).to(self.device)

        # The 'target' network is a delayed copy of the online network.
        # Its weights are frozen and only updated every 'target_update_freq' episodes.
        # This provides stable Q-value targets during training.
        self.target_net = DQN(state_size, action_size, hidden_size).to(self.device)

        # Initialise target network with the same weights as the online network.
        self.target_net.load_state_dict(self.online_net.state_dict())

        # The target is network is never directly trained - gradients are disabled
        self.target_net.eval()

        # --- Optimiser ---
        # Adam is a popular optimiser that adapts the learning rate automatically.
        # It updates the weights of the online network during training.
        self.optimiser = optim.Adam(self.online_net.parameters(), lr = lr)

        # --- Replay Buffer ---
        self.memory = ReplayBuffer(buffer_capacity)

    # ----------------------------------------------------------------------------------
    # ACTION SELECTION
    # ----------------------------------------------------------------------------------
    def select_action(self, state, env = None) -> int:
        """
        Select an action using the epsilon-greedy policy.
        
        With probability epsilon -> pick a random action (exploration).
        With probability 1-epsilon -> pick the best action from the network (exploitation).

        Early in training, epsilon is high so the agent explores a lot.
        Over time, epsilon decays and the agent exploits its learn knowledge more.

        Args:
            state: The current environment state (numpy array of 4 values).

        Returns:
            int: The chosen action (0 = left, 1 = right).
        """
        # --- Exploration: random action ---
        # Use env.action_space.sample() when available — it guarantees
        # the correct type and range expected by Gymnasium.
        if random.random() < self.epsilon:
            if env is not None:
                return int(env.action_space.sample())
            return int(random.randrange(self.action_size))
        
        # --- Exploitation: best action from the network
        # Convert state to tensor and add a batch dimension: (4, ) -> (1, 4)
        state_tensor = torch.tensor(np.array(state, dtype = np.float32)).unsqueeze(0)

        # Disable gradient computation - we're only doing inference, not training
        with torch.no_grad():
            q_values = self.online_net(state_tensor) # shape: (1, action_size)

        # Debug
        #print(f"state shape: {state_tensor.shape}, q_values shape: {q_values.shape}, q_values: {q_values}")

        # Pick the action with the highest Q-value
        return int(q_values.argmax(dim = 1).item())
    
    # ----------------------------------------------------------------------------------
    # LEARNING
    # ----------------------------------------------------------------------------------
    def learn(self):
        """
        Sample a batch from the Replay Buffer and perform one training step.
        
        This is the core of the DQN algorithm. The network learns by minimising the 
        difference between:
            - what it currently predicts (current Q-values)
            - what it should predict (targer Q-values from the Bellman equation)
        """
        # Only start learning once the buffer has enough experiences
        if len(self.memory) < self.batch_size:
            return
        
        # --- Sample a random batch from the Replay Buffer ---
        states, actions, rewards, next_states, done = self.memory.sample(self.batch_size)

        # Move all tensors to the current device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        done = done.to(self.device)

        # --- Current Q-values ---
        # online_net(states) -> shape: (batch_size, action_size)
        # .gather(1, actions.unsqueeze(1)) -> picks the Q-value for the action actually taken
        # Result shape: (batch_size, 1) -> squeeze to (batch_size,)
        actions   = actions.long()  # Ensure int64 dtype required by gather()
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Target Q-values ---
        # Uses Bellman equation;
        #   Q_target = reward + gamma * max(Q_target_net(next_state)) * (1 - done)
        # (1 - done) ensures that if the episode ended (done = 1), the future reward is 0.
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim = 1).values
            target_q = rewards + self.gamma * max_next_q * (1 - done)

        # --- Compute loss ---
        # MSE loss measures how far current predictions are from the targets.
        loss = nn.MSELoss()(current_q, target_q)

        # --- Backpropagation ---
        self.optimiser.zero_grad() # Clears gradient from the previous step
        loss.backward() # Computes gradients with backpropagation
        self.optimiser.step() # Update the online network weights

    # ----------------------------------------------------------------------------------
    # EPSILON DECAY
    # ----------------------------------------------------------------------------------
    def decay_epsilon(self):
        """
        Reduce epsilon after each episode.
        
        Exploration decreases over time as the agent becomes more competent. Epsilon is never 
        allowed to go below epsilon_min.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ----------------------------------------------------------------------------------
    # TARGET NETWORK UPDATE
    # ----------------------------------------------------------------------------------
    def update_target_network(self):
        """
        Copy the weights from the online network to the target network.
        
        Called every 'target_update_freq' episodes.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ----------------------------------------------------------------------------------
    # SAVE / LOAD
    # ----------------------------------------------------------------------------------
    def save(self, path: str):
        """
        Save the online network weights to a file.
        
        Args:
            path (str): File path to save.
        """
        torch.save(self.online_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Load previously saved weights into the online network.
        
        Args:
            path (str): File path to load from.
        """
        self.online_net.load_state_dict(torch.load(path, map_location = self.device))
        print(f"Model loaded from {path}")