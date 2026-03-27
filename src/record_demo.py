"""
Record a GIF of the trained DQN agent playing CartPole.

Usage:
    cd src
    python record_demo.py
    
Output:
    assets/demo.gif
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from PIL import Image

from agent import DQNAgent

# ----------------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------------
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL_PATH  = os.path.join(BASE_DIR, "results", "dqn-cartpole.pth")
GIF_PATH    = os.path.join(BASE_DIR, "assets", "demo.gif")

# ----------------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------------
N_EPISODES  = 3     # Number of episodes to record
FPS         = 30    # Frames per second in the GIF
MAX_STEPS   = 500   # Max steps per episode

def record_demo():
    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train.py first!")
        return
    
    os.makedirs(os.path.join(BASE_DIR, "assets"), exist_ok = True)

    # Load the trained agent with epsilon = 0 (pure exploitation - no random actions)
    agent = DQNAgent(state_size=4, action_size=2, hidden_size=128, epsilon=0.0)
    agent.load(MODEL_PATH)

    # Create the environment in rgb_array mode
    # Returns pixel frames instead of opening a window.
    env = gym.make("CartPole-v1", render_mode = "rgb_array")

    all_frames = []
    episode_rewards = []

    print(f"Recording {N_EPISODES} episodes...")

    for ep in range(1, N_EPISODES + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward: float = 0.0
        ep_frames = []

        for _ in range(MAX_STEPS):
            # Capture the current frame as an RGB numpy array
            frame = env.render()
            if frame is not None:
                ep_frames.append(Image.fromarray(np.array(frame, dtype=np.uint8)))

            action = agent.select_action(state, env)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            state = np.array(next_state, dtype=np.float32)

            if terminated or truncated:
                # Capture one last frame after the episode ends
                last_frame = env.render()
                if last_frame is not None:
                    ep_frames.append(Image.fromarray(np.array(last_frame, dtype=np.uint8)))
                break

        all_frames.extend(ep_frames)
        episode_rewards.append(total_reward)
        print(f"    Episode {ep}: {total_reward:.0f} steps recorded ({len(ep_frames)} frames)")

    env.close()

    # Add a short pause at the end of each episode by duplicating the last frame
    # This is done by inserting extra copies of the last frame between episodes
    duration_ms = int(1000 / FPS) # ms per frame at target FPS

    print(f"\nAssembling GIF ({len(all_frames)} frames at {FPS} fps...)")

    # Save as GIF using Pillow
    all_frames[0].save(
        GIF_PATH,
        save_all        = True,
        append_images   = all_frames[1:],
        duration        = duration_ms,
        loop            = 0, # Loop forever
        optimize        = True
    )

    size_kb = os.path.getsize(GIF_PATH) / 1024
    print(f"GIF saved to {GIF_PATH}\n Size:({size_kb:.0f} KB)")
    print(f"Average reward: {np.mean(episode_rewards):.1f}")

if __name__ == "__main__":
    record_demo()