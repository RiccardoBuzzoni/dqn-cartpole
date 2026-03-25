import random
from collections import deque
from typing import Tuple

import numpy as np
import torch

class ReplayBuffer:
    """
    Experience Replay Buffer.
    
    During training the agent collects 'experiences' in the form of:
        (state, action, reward, next_state, done)
    These are called transitions, and they're stored here.
    
    The agent samples a random batch of past experiences to learn from.
    This method is called 'Experience Replay' and it has two key benefits:
        1. STABILITY: breaks the correlation between consecutive experiences.
        2. EFFICIENCY: each experience can be reused for training.
    """
    def __init__(self, capacity: int):
        """
        Initialises the Replay Buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store. (FIFO)
        """
        # deque is a double-ended queue from Python's standard library.
        self.buffer = deque(maxlen = capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Saves a single transition into the buffer.
        
        Args:
            state: The environment state before the action is taken.
            action (int): The action the agent chooses to perform (0 = left, 1 = right).
            reward (float): The reward received after taking an action. (+1 for each step survived)
            next_state: The environment state after an action is taken.
            done (bool): Whether the episode ended after this step.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples a batch of transitions from the buffer.
        
        Random sampling is the key to breaking the temporal correlation between experiences:
        the agent learns from a diverse mix of past situations, rather than only the most recent.
        
        Args:
            batch_size (int): Number of transitions to sample.
            
        Returns:
            A tuple of 5 tensors, one for each component of the transition:
                (state, action, reward, next_state, done)
            Each tensor has shape [batch_size, ...].
        """
        transitions = random.sample(self.buffer, batch_size)

        # 'transitions' is a list of tuples.
        # zip(*transitions) unpacks the list and groups the tuples by components (zip(*iterable) -> revese zipping):
        # -> (s1, s2, ...), (a1, a1, ...), (r1, r2, ...), ...
        states, actions, rewards, next_states, done = zip(*transitions)

        # Convert each group into a PyTorch tensor for batch computation.
        # float32 is the standard precision for neural network inputs.
        return(
            torch.tensor(np.array(states),      dtype = torch.float32),
            torch.tensor(np.array(actions, dtype = np.int64),     dtype = torch.int64),
            torch.tensor(np.array(rewards),     dtype = torch.float32),
            torch.tensor(np.array(next_states), dtype = torch.float32),
            torch.tensor(np.array(done),        dtype = torch.float32),
        )
    
    def __len__(self) -> int:
        """
        Returns the current number of transitions stored in the buffer.
        This allows using len(replay_buffer) naturally in the code.
        """
        return len(self.buffer)