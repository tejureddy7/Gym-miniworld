import tensorflow as tf
import gym
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DQN

# Define the gym environment
env = make_vec_env('CartPole-v1', n_envs=1)

# Define the MLP policy
policy = MlpPolicy

# Define the model architecture
n_actions = env.action_space.n
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_actions, activation=None)
])

# Create the DQN agent
agent = DQN(policy=policy, env=env, model=model, learning_rate=0.001, buffer_size=5000,
            exploration_fraction=0.1, exploration_final_eps=0.02, target_update_interval=500,
            train_freq=4, batch_size=32, double_q=True, verbose=1)

# Train the agent
agent.learn(total_timesteps=int(1e5))

# Save the agent
agent.save("dqn_mlp_model")

# Evaluate the agent
obs = env.reset()
for i in range(1000):
    action, _states = agent.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Close the environment
env.close()
