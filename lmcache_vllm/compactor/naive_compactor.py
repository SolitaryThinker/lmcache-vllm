import abc
from typing import Tuple, List, Dict



class DropMidCompactor(BaseCompactor):
    """
    SteamingLLM-like compactor
    Always retain the first 4 tokens (attention sinks)
    """
    def __init__(self,):
        self.num_sink = 4
        self.compact_ratio = 0.7 
        
    def compute_indices(self, token_indices):
        """
        
        """
        num_last = int(self.compact_ratio * len(token_indices)) - 4
        new_token_indices = token_indices[:self.num_sink] + \
            token_indices[-num_last:]
        return new_token_indices