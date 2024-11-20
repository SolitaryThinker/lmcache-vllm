import abc
from typing import Tuple, List, Dict
import torch
import queue

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.backends.utils import compute_slot_mapping
from vllm import _custom_ops as ops

from lmcache_vllm.compactor.base_local_compactor import BaseLocalCompactor
from lmcache_vllm.compactor.utils import CompactorOutput
from lmcache.logging import init_logger


class SinkCompactor(BaseLocalCompactor):
    """
    SteamingLLM-like compactor
    Always retain the first 4 tokens (attention sinks)
    """
    def __init__(self,):
        super().__init__()
        
        self.min_window_size = 350
        self.max_window_size = 512
        self.num_sink = 4
    
    def update_imp_scores(
        self,
        seq_id,
        idx,
        chunked_attetnion_weights):
        """
        No `imp_scores` for AttentionSink
        Do nothing
        """
        pass
        
        
    
    def compute_indices(self, seq_id):
        """
        
        """
        num_last = self.min_window_size - self.num_sink
        
        sink_indices = [i for i in range(self.num_sink)]
        last_indices = [i for i in range(self.max_window_size - num_last,
                                         self.max_window_size)]
        compacted_indices = [sink_indices + last_indices \
            for i in range(self.num_layers)]
        
        return compacted_indices