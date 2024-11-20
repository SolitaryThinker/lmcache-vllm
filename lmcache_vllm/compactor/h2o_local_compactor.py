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

logger = init_logger(__name__)

class H2OCompactor(BaseLocalCompactor):
    """
    H2O compactor
    """
    
    def __init__(self):
        super().__init__()
        
        self.min_window_size = 400
        self.max_window_size = 512
            
        
    def update_imp_scores(
        self,
        seq_id,
        idx,
        chunked_attetnion_weights):
        """
        Simply add the attention_weight to the existing imp_scores
        """
        
        for layer_idx in range(self.num_layers):
            attn_weight = chunked_attetnion_weights[layer_idx][idx]
            seq_len = attn_weight.shape[1]
            self.imp_scores[seq_id][layer_idx,:,:seq_len] += \
                attn_weight[layer_idx][idx]
        

    
    def compute_indices(
        self,
        seq_id,
    ):
        """
        compute indices for schedulers
        compact imp_scores
        """
        compacted_indices = []
        imp_score = self.imp_scores[seq_id]
        for layer_idx in range(self.num_layers):
            # sum of all heads
            sum_scores_layer = torch.sum(imp_score[layer_idx], dim=0)
            imp_indices_layer = torch.topk(
                sum_scores_layer, k=self.min_window_size).indices
            
            # TODO: please get rid of this `tolist`
            imp_indices_layer = imp_indices_layer.tolist()
            compacted_indices.append(imp_indices_layer)

            # compact imp_scores
            imp_score[layer_idx,: , :self.min_window_size] = \
                imp_score[layer_idx, :, imp_indices_layer]
            imp_score[layer_idx,: , self.min_window_size:] = 0
        
        return compacted_indices