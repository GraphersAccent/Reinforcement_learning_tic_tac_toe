import numpy as np
import weakref
import gc

class MCTSNode:
    def __init__(self, state, player=None, parent=None, action_taken=None):
        self.state = state
        self.player = player
        self.parent = weakref.ref(parent) if parent is not None else None
        self.children = []
        self.action_taken = action_taken
        self.visits = 0
        self.total_value = 0
        self.untried_actions = self.get_valid_actions()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_node()
        gc.collect()
    
    def get_valid_actions(self):
        occupancy = self.state[:, :, 0] + self.state[:, :, 2]
        mask = (occupancy == 0).astype(np.int32).flatten()
        return np.where(mask == 1)[0]
    
    def remove_untried_action(self, action):
        self.untried_actions = np.delete(self.untried_actions, np.where(self.untried_actions == action))

    def q_value(self, player):
        answer = self.total_value / self.visits if self.visits > 0 else 0
        if self.player != player:
            answer = answer * -1
        return answer
    
    def end_node(self):
        for child in self.children:
            child.end_node()
        
        self.visits = None
        self.total_value = None
        self.untried_actions = None
        self.children.clear()
        self.parent = None
        self.action_taken = None
        self.state = None
        self.player = None

        del self
