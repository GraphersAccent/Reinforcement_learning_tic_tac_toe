import numpy as np
import multiprocessing as mp
import sys

def worker_process(worker_id, experience_queue, episodes=100):
    from Tick_tack_toe_env import TicTacToeEnv
    from Tick_tack_toe_agent import Tick_tack_toe_agent
    from tqdm import trange, tqdm
    import signal
    import os
    
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    env = TicTacToeEnv(simulations=100)
    agent = Tick_tack_toe_agent(env.observation_space, env.action_space.n, use_gpu=False)

    weight_path = "./shared_weights.h5"
    last_check = 0
    check_interval = 5
    max_steps = 9
    
    for episode in range(episodes):
        if episode - last_check >= check_interval:
            if os.path.exists(weight_path):
                try:
                    agent.model.load_weights(weight_path)
                    tqdm.write(f"[Worker {worker_id}] Loaded latest weights at episode {episode}")
                except Exception as e:
                    tqdm.write(f"[Worker {worker_id}] Failed to load weights: {e}")
            last_check = episode

        state, _ = env.reset()
        step = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and step < max_steps:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            experience_queue.put((f"{worker_id}_{episode}", step, state, action, reward, next_state, terminated, truncated))
            state = next_state

def learner_process(experience_queue, stop_event, total_episodes=1000, target_update_freq=20, batch_size=128):
    
    from Tick_tack_toe_env import TicTacToeEnv
    from Tick_tack_toe_agent import Tick_tack_toe_agent
    from datetime import datetime
    from tqdm import trange
    import csv
    import os
    
    env = TicTacToeEnv(simulations=100)
    agent = Tick_tack_toe_agent(env.observation_space, env.action_space.n, batch_size=batch_size)

    path = "./logs"
    os.makedirs(path, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = os.path.join(path, f"training_log_{current_time}.csv")
    
    reward_history = []
    exploration_per_episode = []
    loss_per_episode = []
    terminated_per_episode = []
    truncated_per_episode = []
    steps_made_in_episodes = []

    with open(log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "step", "state", "action", "reward", "terminated", "truncated"])
        writer.writeheader()

        for episode in trange(total_episodes):
            episode_experiences = []
            total_reward = 0
            terminated = False
            truncated = False

            try:
                exp = experience_queue.get(timeout=5)
            except:
                if stop_event.is_set():
                    break
                continue
            
            episode_id, step, state, action, reward, next_state, terminated, truncated = exp
            total_reward += reward
            agent.remember(state, action, reward, next_state, terminated, truncated)
            episode_experiences.append(exp)

            writer.writerow({
                "episode": episode_id,
                "step": step,
                "state": state.tolist(),
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated
            })

            agent.replay()

            if episode % target_update_freq == 0:
                agent.update_target_model()

            reward_history.append(total_reward)
            exploration_per_episode.append(agent.exploration_rate)
            loss_per_episode.append(np.mean(agent.loss_history[-step:]))
            terminated_per_episode.append(terminated)
            truncated_per_episode.append(truncated)
            steps_made_in_episodes.append(step)

    # Final summary or save
    agent.save_models("./output_models", f"tic_tac_toe_{current_time}")
    print("Training complete.")
    
if __name__ == "__main__":
    num_workers = 2
    episodes_per_worker = 200
    total_experiences = num_workers * episodes_per_worker * 9

    ctx = mp.get_context("spawn")
    experience_queue = ctx.Queue(maxsize=1000)

    workers = [
        ctx.Process(target=worker_process, args=(i, experience_queue, episodes_per_worker))
        for i in range(num_workers)
    ]
    learner = ctx.Process(target=learner_process, args=(experience_queue, total_experiences))

    try:
        learner.start()
        for w in workers:
            w.start()

        for w in workers:
            w.join()

        learner.join()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt caught! Terminating all processes...")
        for w in workers:
            w.terminate()
        learner.terminate()
        for w in workers:
            w.join()
        learner.join()
        sys.exit(0)
    except Exception as e:
        print(e)