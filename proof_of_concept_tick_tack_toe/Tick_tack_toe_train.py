import numpy as np
from Tick_tack_toe_env import TicTacToeEnv
from Tick_tack_toe_agent import Tick_tack_toe_agent
from datetime import datetime
from tqdm import trange, tqdm

enviremnt = TicTacToeEnv(simulations=100)

episodes = 400
target_update_freq =  10
max_rounds_in_episode = 9

print("training shape: ", enviremnt.observation_space)

agent = Tick_tack_toe_agent(enviremnt.observation_space, enviremnt.action_space.n)
reward_history = []

for episode in trange(episodes):
    state, info = enviremnt.reset()
    terminated = False
    truncated = False
    total_reward = 0
    steps_in_episode = 0

    while (not terminated and not truncated) and steps_in_episode < max_rounds_in_episode:
        action = agent.act(state)

        next_state, reward, terminated, truncated, info = enviremnt.step(action)
        enviremnt.render()
        
        total_reward += reward

        if truncated:
            reward = -100

        reward += steps_in_episode * 0.01

        agent.remember(state, action, reward, next_state, terminated, truncated)
        
        agent.replay()

        steps_in_episode += 1
        state = next_state

    if episode % target_update_freq == 0:
        agent.update_target_model()
        
    reward_history.append(total_reward)

    tqdm.write(f"Episode {episode + 1}/{episodes}:\n\t- Reward: {total_reward:.3f}\n\t- "
        f"Exploration: {agent.exploration_rate:.3f}\n\t- average loss: {np.mean(agent.loss_history[-5:]):.3f}\n\t- "
        f"average reward: {np.mean(reward_history[-5:]):.3f}\n\t- "
        f"terminated: {terminated} - truncated: {truncated}")

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
agent.save_models("./output_models", f"tic_tac_too_{current_time}")