import gym
from gym import spaces
import numpy as np
import pygame
import random


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, simulations=100, loss_reward=-1.0, win_reward=1.0, draw_reward=0.8):
        super(TicTacToeEnv, self).__init__()
        self.current_player = 0 # 0 for player X, 1 for player O
        self.action_space = spaces.Discrete(9)
        self.done = False
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 5), dtype=np.int32)
        self.state = np.zeros((3, 3, 5), dtype=np.float32)
        # based on the observation space, we can have 5 channels:
        # 0-1:player X history placements (only hold X placements)
        # 2-3:player O history placements (only hold O placements)
        # 4: current player; 0 for X, 1 for O
        # this is what I read in the documentation of the miniGo model, and they had done extensive research on this, incuding my previouse method
        
        self.simulations = simulations
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 100
        self.window_size = self.cell_size * 3

        self.loss_reward = loss_reward
        self.win_reward = win_reward
        self.draw_reward = draw_reward

        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

    def reset(self):
        """Resets the environment to the initial state."""
        super().reset()
        self.state = np.zeros((3, 3, 5), dtype=np.float32)
        self.done = False
        self.current_player = 0
        self.state[:, :, 4] = self.current_player
        return (self.state, {})
    
    def _update_observation(self, state, player, row, col):
        """Updates the observation with the current state and player."""
        new_state = state.copy()
        if player == 0:
            new_state[:, :, 1] = new_state[:, :, 0]
            new_state[row, col, 0] = 1
        else:
            new_state[:, :, 3] = new_state[:, :, 2]
            new_state[row, col, 2] = 1
        return new_state
    
    def _switch_player(self, player):
        """Switches the current player."""
        if player == 0:
            player = 1
        else:
            player = 0
        return player

    def step(self, action):
        """Performs a step in the environment, given an action."""
        if self.done:
            raise ValueError("Game is over. Please reset.")

        row, col = divmod(action, 3)
        
        # check if action is inside the action space and if the cell is empty
        is_valid_action = self.action_space.contains(action) and self._is_legal(self.state, action)
        
        # check if the move was ligal or if it wasn't allowed
        if not is_valid_action:
            self.current_player = self._switch_player(self.current_player)
            self.done = True
            return (self.state, -1, False, True, {"illegal_move": True})
        
        # check if there was a finishing move and the model made a educate choice
        reward_modefier = self._check_if_finishing_move(self.state, row, col)

        self.state = self._update_observation(self.state, self.current_player, row, col)
        
        terminated, winner = self._check_game_over(self.state, self.current_player)
        self.done = terminated
        if terminated:
            if winner == self.current_player:
                reward = self.win_reward # win
            elif winner == -1:
                reward = self.draw_reward # draw
            else:
                reward = self.loss_reward # loss, mostly not possable when you are oplaying
            
        else:
            reward = self._monte_carlo_simulation(self.state, self.current_player) * reward_modefier
        
        self.current_player = self._switch_player(self.current_player)
        self.state[:, :, 4] = self.current_player
        
        if self._board_full(self.state):
            self.done = True
            return (self.state, self.draw_reward, True, False, {"draw": True})
        else:
            return (self.state, reward, terminated, False, {})
    
    
    def _board_full(self, board):
        return np.all((board[:, :, 0] + board[:, :, 2]) >= 1)
    
    
    def render(self):
        """Renders the current state of the board."""
        if self.render_mode != "human":
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
                if self.state[row, col, 0] == 1:
                    pygame.draw.line(surface, (255, 0, 0), (center[0] - 20, center[1] - 20),
                                        (center[0] + 20, center[1] + 20), 4)
                    pygame.draw.line(surface, (255, 0, 0), (center[0] - 20, center[1] + 20),
                                        (center[0] + 20, center[1] - 20), 4)
                if self.state[row, col, 2] == 1:
                    pygame.draw.circle(surface, (0, 0, 255), center, 25, 4)

        if self.render_mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
    
    def close(self):
        if self.window:
            pygame.quit()

    def _check_if_finishing_move(self, temp_board, row, col):
        # check if move would have won the game.
        board = temp_board.copy()
        board = self._update_observation(board, self.current_player, row, col)
        won, _ = self._check_game_over(board, self.current_player)

        # if player hasn't won with the current move check if it either stopped an enemy win or could have won if another move was taken.
        if not won:
            # check every other unpopulated space to see if it could have won
            for i in range(3):
                for j in range(3):
                    board = temp_board.copy()
                    player_choosen_spot = (i == row and j == col)
                    if (board[i, j, 0]  == 0 and board[i, j, 2] == 0) and player_choosen_spot != True:
                        board = self._update_observation(board, self.current_player, i, j)
                        won, _ = self._check_game_over(board, self.current_player)
                        if won:
                            return -1.0

            player = self._switch_player(self.current_player)
            # check if player blocked the enemy from winning.
            board = temp_board.copy()
            board = self._update_observation(board, player, row, col)
            won, winner = self._check_game_over(board, player)
            if won and winner != self.current_player:
                return 1.0
            
            #check if the player could have blocked the enemy from winning.
            for i in range(3):
                for j in range(3):
                    board = temp_board.copy()
                    player_choosen_spot = (i == row and j == col)
                    if (board[i, j, 0]  == 0 and board[i, j, 2] == 0) and player_choosen_spot != True:
                        board = self._update_observation(board, player, i, j)
                        won, winner = self._check_game_over(board, player)
                        if won and winner != self.current_player:
                            return -0.8
        
        return 1.0 # the base nothing is wrong

    def _check_game_over(self, temp_board, player):
        """Checks if the game is over (either win or draw)."""
        channel = 0 if player == 0 else 2
        board = temp_board[:, :, channel]
        for i in range(3):
            if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
                return True, player
        if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
            return True, player
        
        return False, -1
    
    def _is_legal(self, board, action):
        row, col = divmod(action, 3)
        return board[row, col, 0] == 0 and board[row, col, 2] == 0

    def _monte_carlo_simulation(self, board, player, gamma = 0.7):
        """Simulates random games from the current state using Monte Carlo to estimate discounted reward."""
        simulations = 0
        total_discounted_reward = 0.0
        
        for _ in range(self.simulations):
            temp_board = board.copy()
            temp_player = player
            discount = 1.0
            cumulative_reward = 0.0

            while True:
                legal_moves = [i for i in range(9) if self._is_legal(temp_board, i)]
                if len(legal_moves) == 0:
                    cumulative_reward += discount * 0.3  # Draw
                    simulations += 1
                    break

                action = random.choice(legal_moves)
                row, col = divmod(action, 3)
                temp_board = self._update_observation(temp_board, temp_player, row, col)

                terminated, winner = self._check_game_over(temp_board, temp_player)
                if terminated:
                    if winner == player:
                        cumulative_reward += self.win_reward * discount  # Win
                    elif winner == -1:
                        cumulative_reward += self.draw_reward * discount  # Draw
                    else:
                        cumulative_reward += self.loss_reward * discount  # Loss
                    simulations += 1
                    break

                discount *= gamma
                temp_player = self._switch_player(temp_player)

            total_discounted_reward += cumulative_reward

        return total_discounted_reward / simulations if simulations > 0 else 0