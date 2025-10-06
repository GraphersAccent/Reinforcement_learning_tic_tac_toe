import gym
from gym import spaces
import numpy as np
from collections import deque
import pygame


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, simulations=100, loss_reward=-1.0, win_reward=1.0, draw_reward=0.25):
        super(TicTacToeEnv, self).__init__()
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1 # 1 for player X, -1 for player O
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3, 5), dtype=np.int32)
        # based on the observation space, we can have 5 channels:
        # 0-1:player X history placements (only hold X placements)
        # 2-3:player O history placements (only hold O placements)
        # 4: current player; 0 for X, 1 for O
        # this is what I read in the documentation of the miniGo model, and they had done extensive research on this, incuding my previouse method
        self.history_size = 2
        self.X_history = deque(maxlen=self.history_size)
        self.O_history = deque(maxlen=self.history_size)
        for i in range(self.history_size):
            self.X_history.append(np.zeros((3, 3), dtype=int))
            self.O_history.append(np.zeros((3, 3), dtype=int))
        
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

    def get_board_for_player(self, player):
        """retuns a 3x3 board only with the current players placements"""
        return_board = np.zeros((3, 3), dtype=int)
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i, j] == player:
                    return_board[i, j] = 1
        
        return return_board

    def render_observation(self):
        """Returns the current observation of the environment."""
        observation = np.zeros((3, 3, 5), dtype=int)
        # Fill the observation with the current state of the board
        observation[:, :, 0] = self.X_history[1]
        observation[:, :, 1] = self.X_history[0]
        observation[:, :, 2] = self.O_history[1]
        observation[:, :, 3] = self.O_history[0]
        observation[:, :, 4] = 0 if self.current_player > 0 else 1
        return observation

    def reset(self):
        """Resets the environment to the initial state."""
        super().reset()
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1 
        for i in range(self.history_size):
            self.X_history.append(np.zeros((3, 3), dtype=int))
            self.O_history.append(np.zeros((3, 3), dtype=int))
        
        return (self.render_observation(), {})
    
    def step(self, action):
        """Performs a step in the environment, given an action."""
        row, col = divmod(action, 3)
        
        # check if action is inside the action space and if the cell is empty
        is_valid_action = self.action_space.contains(action) and self.board[row, col] == 0
        
        # check if the move was ligal or if it wasn't allowed
        if not is_valid_action:
            self.current_player = -self.current_player
            return (self.render_observation(), -1, False, True, {})
        
        

        self.board[row, col] = self.current_player
        
        # update the history of the players
        history = self.get_board_for_player(self.current_player)
        if self.current_player > 0:
            self.X_history.append(history)
        else:
            self.O_history.append(history)
        
        terminated, winner = self._check_game_over(self.board)
        if terminated:
            if winner == self.current_player:
                reward = self.win_reward # win
            elif winner == 0:
                reward = self.draw_reward # draw
            else:
                reward = self.loss_reward # loss, mostly not possable when you are oplaying
            
        else:
            reward = self._monte_carlo_simulation(self.board, self.current_player)
        
        self.current_player = -self.current_player

        return (self.render_observation(), reward, terminated, False, {})
    
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
                if self.board[row, col] == 1:
                    pygame.draw.line(surface, (255, 0, 0), (center[0] - 20, center[1] - 20),
                                        (center[0] + 20, center[1] + 20), 4)
                    pygame.draw.line(surface, (255, 0, 0), (center[0] - 20, center[1] + 20),
                                        (center[0] + 20, center[1] - 20), 4)
                elif self.board[row, col] == -1:
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

    def _check_if_finishing_move_created(self, temp_board, row, col):
        # check if move would have won the game.
        board = temp_board.copy()
        board[row,col] = self.current_player
        won, _ = self._check_game_over(board)

        multiplier = 1.0

        # if player hasn't won with the current move check if it either stopped an enemy win or could have won if another move was taken.
        if not won:
            # check every other unpopulated space to see if it could have won
            for i in range(3):
                for j in range(3):
                    board = temp_board.copy()
                    player_choosen_spot = (i == row and j == col)
                    if board[i,j] == 0 and player_choosen_spot != True:
                        board[i,j] = self.current_player
                        won, _ = self._check_game_over(board)
                        if won:
                            multiplier -= 1

            # check if player blocked the enemy from winning.
            board = temp_board.copy()
            board[row,col] = -self.current_player
            won, winner = self._check_game_over(board)
            if won and winner != self.current_player:
                multiplier += 1.5
            
            #check if the player could have blocked the enemy from winning.
            for i in range(3):
                for j in range(3):
                    board = temp_board.copy()
                    player_choosen_spot = (i == row and j == col)
                    if board[i,j] == 0 and player_choosen_spot != True:
                        board[i,j] = -self.current_player
                        won, winner = self._check_game_over(board)
                        if won and winner != self.current_player:
                            multiplier -= 1
        elif won and _ == self.current_player:
            multiplier += 2.0

        return multiplier if multiplier > 0 else 0.0

    def _check_game_over(self, temp_board):
        """Checks if the game is over (either win or draw)."""
        for i in range(3):
            if np.abs(temp_board[i, :].sum()) == 3:
                return True, temp_board[i, 0]
            if np.abs(temp_board[:, i].sum()) == 3:
                return True, temp_board[0, i]
        
        if np.abs(temp_board.diagonal().sum()) == 3:
            return True, temp_board[0, 0]
        if np.abs(np.fliplr(temp_board).diagonal().sum()) == 3:
            return True, temp_board[0, 2]
        
        if np.all(temp_board != 0):
            return True, 0
        
        return False, 0

    def _monte_carlo_simulation(self, board, player, gamma = 0.65):
        """Simulates random games from the current state using Monte Carlo to estimate discounted reward."""
        simulations = 0
        total_discounted_reward = 0.0
        
        for _ in range(self.simulations):
            temp_board = board.copy()
            temp_player = player
            discount = 1.0
            cumulative_reward = 0.0
            reward_modefier = []

            while True:
                available_moves = np.argwhere(temp_board == 0)
                if len(available_moves) == 0:
                    cumulative_reward += discount * 0.3  # Draw
                    simulations += 1
                    break

                # check if there was a finishing move and the model made a educate choice

                move = available_moves[np.random.randint(0, len(available_moves))]
                row, col = move[0], move[1]

                reward_modefier.append(self._check_if_finishing_move_created(temp_board, row, col))

                temp_board[row, col] = temp_player

                terminated, winner = self._check_game_over(temp_board)
                if terminated:
                    reward_modefier = np.mean(reward_modefier) if reward_modefier else 1.0
                    reward_modefier = max(reward_modefier, 0.1)
                    if winner == player:
                        cumulative_reward += (self.win_reward * reward_modefier) * discount  # Win
                    elif winner == 0:
                        cumulative_reward += (self.draw_reward * reward_modefier) * discount  # Draw
                    else:
                        cumulative_reward += self.loss_reward * discount  # Loss
                    simulations += 1
                    break

                discount *= gamma
                temp_player = -temp_player

            total_discounted_reward += cumulative_reward

        return total_discounted_reward / simulations if simulations > 0 else 0