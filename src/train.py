import os
import sys
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

# Make sure Python can find the other modules in the src/ folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import DQNAgent

# ----------------------------------------------------------------------------------
# HYPERPARAMETERS
# ----------------------------------------------------------------------------------
# Environment
ENV_NAME        = "CartPole-v1"     # Gymnasium env to train on
MAX_EPISODES    = 600               # Total number of episodes to train for
MAX_STEPS       = 500               # Max steps per episode

# Agent
STATE_SIZE      = 4                 # CartPole observation space [pos, vel, angle, ang_vel]
ACTION_SIZE     = 2                 # CartPole action space [left, right]
HIDDEN_SIZE     = 128               # Neurons in each hidden layer of the DQN
LR              = 1e-3              # Learning rate for Adam optimiser
GAMMA           = 0.99              # Discount factor for future rewards
EPSILON         = 1.0               # Initial exploration rate (100% random)
EPSILON_MIN     = 0.01              # Minimum exploration rate
EPSILON_DECAY   = 0.995             # Multiplicative decay per episode
BUFFER_CAPACITY = 10_000            # Max experiences stored in Replay Buffer
BATCH_SIZE      = 64                # Experiences sampled per training step
TARGET_UPDATE   = 10                # Experiences between each target network update

# Saving
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MODEL_PATH      = os.path.join(RESULTS_DIR, "dqn-cartpole.pth")
PLOT_PATH       = os.path.join(RESULTS_DIR, "training_curve.png")

# Logging
PRINT_EVERY     = 20                # Print progress every PRINT_EVERY steps
SOLVE_SCORE     = 475               # CartPole is considered "solved" at this average score

# ----------------------------------------------------------------------------------
# PLOTTING
# ----------------------------------------------------------------------------------
def plot_training(episode_rewards: list, avg_rewards: list, save_path: str):
    """
    Plot and save training curve.
    
    Shows two lines:
        - Episode reward: raw score for each individual episode (noisy)
        - Average reward: rolling average over the last 50 episodes (smooth trend)
    
    Args:
        episode_rewards (list): Raw reward for each episode
        avg_rewards (list): Rolling average reward for each episode
        save_path (str): Where to save the resulting PNG file
    """
    plt.figure(figsize = (12, 5))

    # Raw per-episode rewards
    plt.plot(episode_rewards, color = "steelblue", alpha = 0.4, linewidth = 0.8, label = "Episode Reward")

    # Rolling average
    plt.plot(avg_rewards, color = "darkorange", linewidth = 2, label = "Average Reward (last 50 episodes)")

    # "Solved" threshold line
    plt.axhline(y = SOLVE_SCORE, color = "green", linestyle = "--", linewidth = 1.2, label = f"Solved ({SOLVE_SCORE})")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training - CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to {save_path}")

# ----------------------------------------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------------------------------------
def train():
    """
    Main training loop.
    
    Each episode:
        1. Reset the environment and get initial state
        2. Loop over steps:
            a. Agent selects an action (epsilon-greedy)
            b. Environment executes said action and returns (next_state, reward, done)
            c. Transition is stored in the Replay Buffer
            d. Agent learns from a random batch of past experiences
        3. After the episode ends:
            a. Decay epsilon (less exploration over time)
            b. Possibly update target network
            c. Log progress and check if the environment was solved
    """
    # Create the result directory if it doesn't yet exist
    os.makedirs(RESULTS_DIR, exist_ok = True)

    # --- Initialise environment and agent ---
    env = gym.make(ENV_NAME)
    agent = DQNAgent(
        state_size          = STATE_SIZE,
        action_size         = ACTION_SIZE,
        hidden_size         = HIDDEN_SIZE,
        lr                  = LR,
        gamma               = GAMMA,
        epsilon             = EPSILON,
        epsilon_min         = EPSILON_MIN,
        epsilon_decay       = EPSILON_DECAY,
        buffer_capacity     = BUFFER_CAPACITY,
        batch_size          = BATCH_SIZE,
        target_update_freq  = TARGET_UPDATE,   
    )

    # --- Tracking ---
    episode_rewards = [] # Raw reward for each episode
    avg_rewards = [] # Rolling average over the las 50 episodes
    best_avg_reward = -np.inf # Ensures that the first reward will always be greater

    print(f"\n{'='*50}")
    print(f"    Environment : {ENV_NAME}")
    print(f"    Episodes    : {MAX_EPISODES}")
    print(f"    Device      : {agent.device}")
    print(f"\n{'='*50}")

    # --- Main loop ---
    for episode in range(1, MAX_EPISODES + 1):

        # Reset the environment at the start of each episode
        # 'state' is a numpy array of 4 values describing the initial observation
        state, _ = env.reset()
        state = np.array(state, dtype = np.float32) # # Ensure consistent float32 numpy array
        total_reward: float = 0.0

        # --- Step loop: runs until the pole falls or max steps is reached ---
        for step in range(MAX_STEPS):

            # 1. Agent picks an action based on the current state
            action = agent.select_action(state, env)
            # Debug
            #assert action in [0, 1], f"Invalid action {action} at step {step}, state={state}, type={type(state)}"

            # 2. Environment executes the action and returns:
            #   - next_state : new observation after the action
            #   - reward : +1.0 for every step the pole doesn't fall
            #   - terminated : True if the pole fell or the cart went out of bounds
            #   - truncated : True if MAX_STEPS was reached
            #   - info : extra dignostic info
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)   # Cast to float to satisfy the type checker
            done = terminated or truncated

            # 3. Store the experience in the Replay Buffer
            agent.memory.push(state, action, reward, next_state, done)

            # 4. Agent learns from a random batch of past episodes
            agent.learn()

            # Move to the next state
            state = np.array(next_state, dtype = np.float32)
            total_reward += reward

            # End the episode if it's done
            if done:
                break
        
        # --- End of episode ---
        # Decay epsilon: the agent explores less and becomes more competent
        agent.decay_epsilon()

        # Periodically sync the target network with the online network
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        # --- Track Rewards ---
        episode_rewards.append(total_reward)
        avg = np.mean(episode_rewards[-50:]) # Rolling average over last 50 episodes
        avg_rewards.append(avg)

        # --- Save the best model so far ---
        if avg > best_avg_reward:
            best_avg_reward = avg
            agent.save(MODEL_PATH)

        # --- Print progress every PRINT_EVERY episodes ---
        if episode % PRINT_EVERY == 0:
            print(
                f"Episode {episode:4d}/{MAX_EPISODES} | "
                f"Reward: {total_reward:6.1f} | "
                f"Avg(50): {avg:6.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
        
        # --- Check if the environment is solved ---
        if avg >= SOLVE_SCORE and episode >= 50:
            print(f"\n✅ Solved at episode {episode} | Avg reward: {avg:.1f}")
            agent.save(MODEL_PATH)
            break
    
    # --- Training complete ---
    env.close()
    plot_training(episode_rewards, avg_rewards, PLOT_PATH)
    print(f"\nTraining complete. Best avg reward: {best_avg_reward:.1f}")

# ----------------------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    train()