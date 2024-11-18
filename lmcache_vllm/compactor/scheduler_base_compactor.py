import abc
from typing import Tuple, List, Dict
import torch

# FIXME(Jiayi): this LocalCompactor design need to be 
# compatible with PP/TP some how
class BaseSchedulerCompactor(metaclass=abc.ABCMeta):
    """
    Interface for scheduler compactor
    """
    
    # NOTE(Jiayi): similar to the 
    @abc.abstractmethod
    def compact_metadata(
        self,
        compacted_indices_dict,
        seq_group: SequenceGroup):
        """
        Perform scheduler metadata compaction here.
        
        Return: kv_mmaps (used for the actual data movement).
        
        """
        raise NotImplementedError
        