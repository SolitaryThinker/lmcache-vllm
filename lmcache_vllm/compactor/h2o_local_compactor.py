import abc
from typing import Tuple, List, Dict
from base_loca_compactor import BaseLocalCompactor
import torch


class H2OCompactor(BaseLocalCompactor):
    """
    H2O compactor
    """
    def __init__(self):
        # NOTE(Jiayi): keeping src_slot_mappings in local compactor
        # minimizes communication overhead between scheduler and worker
        # However, the following case would result a memory leak:
        # local compactor decides to compact but scheduler finishes
        # the req.
         
        #{seq_idx: num_layers * slot_mapping}
        self.src_slot_mappings = {}
        
        # tensor: num_layer, num_head, window_limit
        #{seq_idx: imp_scores}
        # imp_scores should be initialized as the seq_id enters
        self.imp_scores = {}
        
        self.min_window_size = 1024
        self.max_window_size = 4096
        
        #num_layer * Tensor([num_heads, num_toks])
        self.buffer_attn_weights = []
        
        # TODO: remove this hardcode
        self.num_layers = 32
    
    def update_imp_scores_buffer(
        self,
        new_attn_weights,
    ):
        self.buffer_attn_weights.append(new_attn_weights)
        
    def update_imp_scores(
        self,
        seq_id,
        chunked_attetnion_weight):
        """
        """
        seq_len = chunked_attetnion_weights.shape[2]
        
        for layer_idx in range(self.num_layers):
            self.imp_scores[seq_id][num_layer,:,:seq_len] += chunked_attetnion_weight
        
            

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
        
        for seq_id, dst_slot_mapping in dst_slot_mappings.items():
            dst_slot_mapping = torch.tensor(dst_slot_mapping, 
                                            device=kv_cache[0].device)
            
            # TODO(Jiayi): Figure out why there are pending 0s in block_tables
            # Might be related to cuda graph & max batch_size
            # https://github.com/vllm-project/vllm/blob/ad23318928d40ef7ac969451afa0dc198428c04b/vllm/attention/backends/flash_attn.py#L370
            
            # TODO(Jiayi): optimize the following code into a cuda kernel?
            # or at least into a separate function
            for layer_idx, src_slot_mapping_layer in \
                enumerate(self.src_slot_mappings[seq_id]):
                
                kv_cache = kv_caches[layer_idx]
                attn_layer = attn_layers[i]
                key_cache, value_cache = kv_cache[0], kv_cache[1]
                _, _, num_heads, head_size = kv_cache[0].shape
                key_cache_temp = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache_temp = kv_cache[1].reshape(-1, num_heads, head_size)
                ops.reshape_and_cache_flash(
                    key_cache_temp[src_slot_mapping],
                    value_cache_temp[src_slot_mapping],
                    key_cache,
                    value_cache,
                    dst_slot_mapping,
                    attn_layer.attn.kv_cache_dtype,
                    attn_layer.attn._k_scale,
                    attn_layer.attn._v_scale,
                )
            
            # pop src_slot_mapping to reduce memory usage
            self.src_slot_mappings.pop(seq_id)
    
    def compute_inidces(
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
            sum_scores_layer = torch.sum(imp_scores[layer_idx], dim=0)
            imp_indices_layer = torch.topk(
                compacted_indices_layer, k=self.min_window_size)
            
            # TODO: please get rid of this `tolist`
            imp_indices_layer = imp_indices_layer.tolist()
            compacted_indices.append(imp_indices_layer)

            # compact imp_scores
            imp_scores[layer_idx,: , :self.min_window_size] = \
                imp_scores[layer_idx, :, imp_indices_layer]
            imp_scores[layer_idx,: , self.min_window_size:] = 0
        
        return compacted_indices
            
    def post_model_update(
        self,
        model_input,
        seq_group_metadata_list):
        """
        1. update imp_scores
        2. Conditionally compute indices for schedulers
        3. Conditionally update src_slot_mapping
        """
        seq_lens = model_input.attn_metadata.seq_lens
        chunked_attetnion_weights = torch.split(
            self.buffer_attn_weights, seq_lens, dim=2)
        compacted_indices_dict = {}
        idx = 0
        for seq_group_metadata in seq_group_metadata_list:
            request_id = seq_group_metadata.request_id
            seq_ids = model_input.request_ids_to_seq_ids[request_id]
            for seq_id in seq_ids:
                self.update_imp_scores(
                    seq_id,
                    chunked_attetnion_weights[:, :, idx]
                )
                seq_data = seq_group_metadata.seq_data[seq_id]
                seq_len = seq_data.get_len()
                
                # FIXME(Jiayi): fix the logic here
                if seq_len < self.max_window_size:
                    break
                
                compacted_indices = self.compute_indices(seq_id)
                compacted_indices_dict[seq_id] = compacted_indices

                # update src_slot_mappings
                slot_mapping = []
                vllm_block_size = 16
                compute_slot_mapping(False, slot_mapping, seq_id, seq_len, 
                    0, 0, vllm_block_size, seq_group_metadata.block_tables)
                
                self.src_slot_mappings[seq_id] = slot_mapping[compacted_indices]
                
        compactor_output = CompactorOutput(
            compacted_indices_dict=compacted_indices_dict,)
        return compactor_output