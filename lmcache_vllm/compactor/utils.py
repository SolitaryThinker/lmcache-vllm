from typing import Tuple, List, Dict
from dataclasses import dataclass
import torch


# TODO(Jiayi): The following assumption needs to be more flexible
# Current assumption: 
# Across layers: same number, different tokens
# Across heads: same number, same tokens

@dataclass
class CompactorInput:
    # map from old slot mapping to new slot mapping
    #kv_mmaps: List[Tuple[List[int], List[int]]]
    
    # dst memory across all heads and layers
    # Since number of tokens are uniform, we can reuse block tables
    # across all layers
    
    # {seq_idx: List[int]}
    dst_slot_mappings: Dict[int, List[int]]


# NOTE(Jiayi): a potential optimization is to only send the
# number of compacted tokens back to scheduler
@dataclass
class CompactorOutput:
    compacted_indices_dict: Dict[int, List[List[int]]]
    end_seq_ids: List[int]

