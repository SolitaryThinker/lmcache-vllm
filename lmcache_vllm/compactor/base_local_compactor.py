import abc
from typing import Tuple, List, Dict
import torch
import queue

from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.backends.utils import compute_slot_mapping
from vllm import _custom_ops as ops

from lmcache_vllm.compactor.utils import CompactorOutput
from lmcache.logging import init_logger

logger = init_logger(__name__)

# FIXME(Jiayi): this LocalCompactor design need to be 
# compatible with PP/TP some how
class BaseLocalCompactor(metaclass=abc.ABCMeta):
    """
    Interface for local compactor
    """
    
    def __init__(self):
        # NOTE(Jiayi): keeping src_slot_mappings in local compactor
        # minimizes communication overhead between scheduler and worker
         
        #{seq_idx: num_layers * slot_mapping}
        self.src_slot_mappings = {}
        
        # tensor: num_layer, num_head, window_limit
        #{seq_idx: imp_scores}
        # imp_scores should be initialized as the seq_id enters
        self.imp_scores = {}
        
        #num_layer * Tensor([num_heads, num_toks])
        
        
        # TODO: remove this hardcode
        self.num_layers = 32
        self.num_heads = 32
        self.num_kv_heads = 8
        self.head_size = 128
        self.device = "cuda"
        self.vllm_block_size = 16

        # The logits buffer need to be preallocated
        # to be compatible with cuda graph
        # TODO: remove hard code `81920`
        # TODO: queue looks weird here. This queue exists only because
        # layer_idx is not available in attention module
        self.logits_buffer_queue = queue.Queue()
        for i in range(self.num_layers):
            self.logits_buffer_queue.put(
                torch.empty((self.num_heads, 81920),
                        device=self.device,
                        dtype=torch.float32)
                )
    
    @abc.abstractmethod
    def update_imp_scores(
        self,
        seq_id,
        idx,
        chunked_attetnion_weights):
        """
        update importance scores
        """
        
        raise NotImplementedError
    
    @abc.abstractmethod
    def compute_indices(
        self,
        seq_id,
    ):
        """
        compute indices for schedulers
        compact imp_scores
        """
        raise NotImplementedError
    
        
    def allocate_imp_scores(
        self,
        model_input,
    ):
        seq_group_metadata_list = model_input.seq_group_metadata_list
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            seq_ids = model_input.request_ids_to_seq_ids[request_id]
            for seq_id in seq_ids:
                if seq_id in self.imp_scores:
                    continue
                imp_scores_temp = torch.zeros(
                    (self.num_layers, self.num_heads, self.max_window_size),
                    device=self.device,
                    dtype=torch.float32)
                self.imp_scores[seq_id] = imp_scores_temp

    def compact_memory(
        self,
        model_input_subset,
        kv_caches,
        dst_slot_mappings):
        """
        """
        attn_layers = model_input_subset.attn_layers
        start_layer = model_input_subset.start_layer
        end_layer = model_input_subset.end_layer
        
        # TODO(Jiayi): intra-batch memory movement should be batched 
        for seq_id, dst_slot_mapping in dst_slot_mappings.items():
            dst_slot_mapping = torch.tensor(dst_slot_mapping, 
                                            device=kv_caches[0][0].device)
            
            # TODO(Jiayi): Figure out why there are pending 0s in block_tables
            # Might be related to cuda graph & max batch_size
            # https://github.com/vllm-project/vllm/blob/ad23318928d40ef7ac969451afa0dc198428c04b/vllm/attention/backends/flash_attn.py#L370
            
            # TODO(Jiayi): optimize the following code into a cuda kernel?
            # or at least into a separate function
            for layer_idx, src_slot_mapping_layer in \
                enumerate(self.src_slot_mappings[seq_id]):
                
                kv_cache = kv_caches[layer_idx]
                attn_layer = attn_layers[layer_idx]
                key_cache, value_cache = PagedAttention.split_kv_cache(
                        kv_cache, self.num_kv_heads, self.head_size)
                
                #import pdb
                #pdb.set_trace()
                key_cache_temp = key_cache.permute(0,3,1,2,4)
                key_cache_temp = key_cache_temp.reshape(
                                -1, self.num_kv_heads, self.head_size)
                
                value_cache_temp = value_cache.permute(0,3,1,2)
                value_cache_temp = value_cache_temp.reshape(
                                -1, self.num_kv_heads, self.head_size)
                
                src_slot_mapping_layer = torch.tensor(
                    src_slot_mapping_layer, 
                    device=dst_slot_mapping.device)
                
                assert len(src_slot_mapping_layer) == len(dst_slot_mapping)
                misaligned_indices = torch.where(
                    src_slot_mapping_layer != dst_slot_mapping)[0]
                
                if len(misaligned_indices) == 0:
                    continue

                # reshape_and_cache_flash is only used for flash attention
                ops.reshape_and_cache(
                    key_cache_temp[src_slot_mapping_layer[misaligned_indices]],
                    value_cache_temp[src_slot_mapping_layer[misaligned_indices]],
                    key_cache,
                    value_cache,
                    dst_slot_mapping[misaligned_indices],
                    attn_layer.attn.kv_cache_dtype,
                    attn_layer.attn._k_scale,
                    attn_layer.attn._v_scale,
                )
            
            # pop src_slot_mapping to reduce memory usage
            self.src_slot_mappings.pop(seq_id, None)
    
    def clean_request_states(
        self,
        end_seq_ids,
    ):
        if end_seq_ids is None:
            return
        for end_seq_id in end_seq_ids:
            self.src_slot_mappings.pop(end_seq_id, None)
            self.imp_scores.pop(end_seq_id, None)
        
    
    def post_model_update(
        self,
        kv_caches,
        model_input):
        """
        1. update imp_scores
        2. Conditionally compute indices for schedulers
        3. Conditionally update src_slot_mapping
        """
        
        # skip profile run
        is_profile_run = (kv_caches is None) or (kv_caches[0] is None)
        if is_profile_run:
            return
        
        seq_group_metadata_list = model_input.seq_group_metadata_list
        attn_meta = model_input.attn_metadata
        prefill_meta = attn_meta.prefill_metadata
        
        seq_lens = attn_meta.seq_lens
        sum_seq_len = sum(seq_lens)
        
        chunked_attetnion_weights = None
        
        # FIXME(Jiayi): we are skipping prefill for now
        is_all_prefill_run = ((attn_meta.num_prefills == len(seq_lens))\
            and prefill_meta is not None)
        
        if is_all_prefill_run:
            null_compactor_output = CompactorOutput(compacted_indices_dict={})
            return null_compactor_output
           
        chunked_attetnion_weights = []
        for i in range(self.num_layers):
            buffer = self.logits_buffer_queue.get()
            chunked_buffer = torch.split(
                buffer[:, :sum_seq_len], 
                seq_lens, dim=1)
            chunked_attetnion_weights.append(chunked_buffer)
            self.logits_buffer_queue.put(buffer)
        
        compacted_indices_dict = {}
        idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            seq_ids = model_input.request_ids_to_seq_ids[request_id]
            for seq_id in seq_ids:
                if chunked_attetnion_weights is not None:
                    self.update_imp_scores(
                        seq_id,
                        idx,
                        chunked_attetnion_weights,
                    )
                seq_data = seq_group_metadata.seq_data[seq_id]
                seq_len = seq_data.get_len()
                
                # FIXME(Jiayi): fix the logic here
                if seq_len < self.max_window_size:
                    continue
                
                logger.debug(f"[Compactor] h2o_local_compactor taking effect! seq_id: {seq_id}")
                compacted_indices = self.compute_indices(seq_id)
                compacted_indices_dict[seq_id] = compacted_indices

                # update src_slot_mappings
                slot_mapping = []
                compute_slot_mapping(False, slot_mapping, seq_id, seq_len, 
                    0, 0, self.vllm_block_size, seq_group_metadata.block_tables)
                
                compacted_slot_mapping =[]
                for compacted_indices_layer in compacted_indices:
                    compacted_slot_mapping.append(
                        [slot_mapping[i] for i in compacted_indices_layer])
                
                self.src_slot_mappings[seq_id] = compacted_slot_mapping
                
        compactor_output = CompactorOutput(
            compacted_indices_dict=compacted_indices_dict,)
        return compactor_output
    

