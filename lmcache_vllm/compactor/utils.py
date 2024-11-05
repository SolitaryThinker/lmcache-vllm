from typing import Tuple, List, Dict
from dataclasses import dataclass
import torch

@dataclass
class CompactorInput:
    # map from old slot mapping to new slot mapping
    kv_mmaps: List[Tuple[List[int], List[int]]]

@dataclass
class CompactorOutput:
    compacted_indices_dict: Dict[int, List[int]]

