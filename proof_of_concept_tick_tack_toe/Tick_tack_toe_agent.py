import gym
import os
from gym import spaces
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
from collections import deque
import random
from WarmUpAndDecay import WarmUpAndDecay
import pygame
import pandas as pd
import json



class Tick_tack_toe_agent:
    metadata = {"render_fps": 10}
    
    def __init__(self, state_shape, n_actions, **params):
        
        self.use_gpu = params.get('use_gpu', True)
        self.device = "/GPU:0" if self.use_gpu else "/CPU:0"
        self.state_shape = state_shape.shape
        self.n_actions = n_actions

        self.alpha = np.ones(n_actions) 
        self.beta = np.ones(n_actions) 

        self.action_counts = np.zeros(n_actions)
        self.total_actions = 0

        self.exploration_rate = params.get('exploration_rate', 1.0)
        self.min_exploration_rate = params.get('min_exploration_rate', 0.1)
        self.exploration_decay = params.get('exploration_decay', 0.995)

        self.gamma = params.get('gamma', 0.99)
        self.batch_size = params.get('batch_size', 64)
        self.memory_size = params.get('memory_size', 2000)
        self.memory = deque([], maxlen=self.memory_size)

        # Load training data
        self.training_data_path = params.get('training_data_path', './tic_tac_toe_training_data/train_data.csv')
        if os.path.exists(self.training_data_path):
            self.train_data = pd.read_csv('./tic_tac_toe_training_data/train_data.csv', sep=';')
            self.train_data['states'] = self.train_data['states'].apply(lambda x: np.array(json.loads(x)))

        self.Human_player = params.get('Human_player', False)
        self.cell_size = params.get('cell_size', 100)
        self.window_size = self.cell_size * 3
        
        self.window = None
        if self.Human_player:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        self.use_pretrained = params.get('use_pretrained', False)
        if not self.use_pretrained:
            self.learning_rate = WarmUpAndDecay(
            base_lr= params.get('learning_rate', 0.001),
            warmup_steps=params.get('warmup_steps', 1000),
            decay_steps=params.get('decay_steps', 10000)
            )

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            self.filter_size =  params.get('filter_size', 16)
            self.num_res_blocks = params.get('num_res_blocks', 2)

            with tf.device(self.device):
                self.model = self._build_model(filter_size=self.filter_size, num_res_blocks=self.num_res_blocks)
                self.target_model = self._build_model(filter_size=self.filter_size, num_res_blocks=self.num_res_blocks)

        else:
            self.model_path = params.get('model_path', 'pretrained_model.h5')
            with tf.device(self.device):
                self.model = self._load_pretrained_model()
                self.target_model = self._load_pretrained_model()

        
        print("Model:", self.model)
        print("Target model:", self.target_model)
        
        self.update_target_model()

        self.reward_history = []
        self.exploration_rate_history = []
        self.episodes = params.get('episodes', 1000)
        self.action_count_history = np.zeros((self.n_actions, self.episodes))
        self.current_episode = 0
        self.loss_history = []
        self.accuracy_history = []
    
    def _load_pretrained_model(self):
        """Load a pretrained model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Pretrained model not found at {self.model_path}")
        
        model = load_model(
        self.model_path,
            custom_objects={
                'WarmUpAndDecay': WarmUpAndDecay
            }
        )

        print(f"Pretrained model loaded from {self.model_path}")
        print("Model summary:")
        model.summary()
        return model

    def _build_model(self, filter_size=16, num_res_blocks=2):
        """Build the Q-network model."""
        print(f"Building model with state shape: {self.state_shape} and action space: {self.n_actions}")
        
        inputs = tf.keras.Input(shape=self.state_shape, dtype=tf.float32)
        
        x = layers.Conv2D(filter_size, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        # create residual blocks to stableze thinking proces
        for _ in range(num_res_blocks):
            residual = x
            x = layers.Conv2D(filter_size, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filter_size, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(2, 1, activation='relu')(x)
        x = layers.Flatten()(x)
        output = layers.Dense(9, activation='softmax', name='policy_head')(x)
        # output = layers.Dense(9, activation='linear')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=output)
        model.summary()
        model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        return model

    def save_models(self, path, name, save_both=True):
        os.makedirs(path, exist_ok=True)

        savedmodel_path = os.path.join(path, name+ '.h5')
        
        self.target_model.save(savedmodel_path)
        print(f"Model saved in HDF5 format at: {savedmodel_path}")

        if save_both:
            h5_path = os.path.join(path, name + '_extra' + '.h5')
            self.model.save(h5_path)
            print(f"Model also saved in HDF5 format at: {h5_path}")

    def update_target_model(self):
        """Update the target model with the weights of the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, terminated, truncated):
        """Store the experience in memory."""
        experience = (state, action, reward, next_state, terminated, truncated)
        self.memory.append(experience)
    
    def act(self, state):
        """Select an action based on the current policy and state."""

        # Use the model to predict Q-values for the current state
        board = tf.convert_to_tensor(state, dtype=tf.float32)
        board = tf.expand_dims(board, axis=0)

        # make sure the q_values are valid
        # 1 for valid, 0 for invalid
        occupancy = board[0, :, :, 0] + board[0, :, :, 2]
        occupancy = occupancy.numpy()
        mask = tf.cast(tf.equal(occupancy, 0), tf.float32)
        flat_mask = tf.reshape(mask, [-1])

        if np.random.rand() < self.exploration_rate:
            flat_mask_np = flat_mask.numpy()
            valid_actions = np.where(flat_mask_np == 1)[0] 

            return np.random.choice(valid_actions)
    
        q_values = self.model.predict(board, verbose=0)[0]
        
        return np.argmax(q_values)

    def replay(self):
        """Train the model using a batch of experiences from memory, that is created by earlier interactions with the environment."""
        if len(self.memory) < self.batch_size:
            return

        experiences = random.sample(population=self.memory, k=self.batch_size)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        terminateds = []
        truncateds = []
        
        for state, action, reward, next_state, terminated, truncated in experiences:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminateds.append(terminated)
            truncateds.append(truncated)

        terminateds = np.array(terminateds)
        truncateds = np.array(truncateds)
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tf = (terminateds | truncateds).astype(np.float32)
        
        target_q_values_next = self.target_model.predict(next_states_tf, verbose=0)
        max_q_next = tf.reduce_max(target_q_values_next, axis=1)
        
        td_targets = rewards_tf + self.gamma * max_q_next * (1.0 - dones_tf)
        
        q_values = self.model.predict(states_tf, verbose=0)
        target_q_values = np.array(q_values)
        
        for idx, action in enumerate(actions):
            target_q_values[idx][action] = td_targets.numpy()[idx]
        

        if self.train_data is not None and len(self.train_data) > self.batch_size * 2:
            extended_training_data = self.train_data.sample(n=self.batch_size * 2)

            ext_moves = pd.get_dummies([row.moves for row in extended_training_data.itertuples()], prefix='pos', dtype=float)

            ext_moves = ext_moves.values.astype(np.float32)
            target_q_values = np.vstack([target_q_values, ext_moves])
            target_q_values = np.array(target_q_values, dtype=np.float32)
            
            states_tf = tf.concat([states_tf, tf.convert_to_tensor([row.states for row in extended_training_data.itertuples()], dtype=tf.float32)], axis=0)
        

        training = self.model.fit(states_tf, target_q_values, epochs=1, verbose=0)
        
        self.loss_history.extend(training.history['loss'])
        self.accuracy_history.extend(training.history['accuracy'])
        
        num_actions = tf.cast(self.n_actions, tf.float32)

        epsilon = self.exploration_rate
        greedy_prob = 1 - epsilon + (epsilon / num_actions)
        non_greedy_prob = epsilon / num_actions
        
        greedy_actions = tf.cast(tf.argmax(target_q_values_next, axis=1), tf.int32)

        expected_next_q = tf.reduce_sum(
            target_q_values_next * (
                tf.one_hot(greedy_actions, self.n_actions) * greedy_prob +
                (1 - tf.one_hot(greedy_actions, self.n_actions)) * non_greedy_prob
            ),
            axis=1
        )   

        target_q_values = rewards_tf + (1.0 - dones_tf) * self.gamma * expected_next_q
        
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)
        
        for i in range(self.batch_size):
            action = actions[i]
            reward = rewards[i]

            self.action_counts[action] += 1
            self.total_actions += 1

            if reward > 0:
                self.alpha[action] += 1
            else:
                self.beta[action] += 1

        self.action_count_history = np.column_stack((self.action_count_history, self.action_counts))
        self.action_count_history[:, self.current_episode] = self.action_counts
        self.current_episode += 1
        
        avg_expected_q = tf.reduce_mean(target_q_values).numpy()
        self.reward_history.append(avg_expected_q)
    
    def start_human_play(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
    
    def stop_human_play(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
    
    def render(self, state=None):
        """Renders the current state of the board."""
        if self.window is None:
            return
        # do event pump to make sure it stays active
        pygame.event.pump()

        #create a white surfice to as a starting of the frame.
        surface = pygame.Surface((self.window_size, self.window_size))
        surface.fill((255, 255, 255))

        # create the 4 lines of tic-tac-toe
        for i in range(1, 3):
            pygame.draw.line(surface, (0, 0, 0), (0, i * self.cell_size), (self.window_size, i * self.cell_size), 3)
            pygame.draw.line(surface, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.window_size), 3)

        # loop through the board and draw a circle or a cross.
        for row in range(3):
            for col in range(3):
                # get the center of the possition we are checking by getting the top right corner of the square, 
                # and then adding half the cell size randed on it to get the senter
                center = (col * self.cell_size + self.cell_size // 2, row * self.cell_size + self.cell_size // 2)
                if state[row, col, 0] == 1:
                    pygame.draw.line(surface, (255, 0, 0), (center[0] - 20, center[1] - 20),
                                        (center[0] + 20, center[1] + 20), 4)
                    pygame.draw.line(surface, (255, 0, 0), (center[0] - 20, center[1] + 20),
                                        (center[0] + 20, center[1] - 20), 4)
                elif state[row, col, 2] == 1:
                    pygame.draw.circle(surface, (0, 0, 255), center, 25, 4)

        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def human_act(self):
        """Wait for human input and return the chosen action."""
        if self.window is None:
            return

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_human_play()
                    return None  # You may want to signal a quit here

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = mouse_y // self.cell_size
                    col = mouse_x // self.cell_size

                    if 0 <= row < 3 and 0 <= col < 3:
                        action = row * 3 + col
                        return action

            pygame.time.wait(10)  # Sleep briefly to avoid high CPU usage
