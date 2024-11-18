import abc
from typing import Tuple, List, Dict
import torch

# FIXME(Jiayi): this LocalCompactor design need to be 
# compatible with PP/TP some how
class BaseSchedulerCompactor:
    """
    Interface for scheduler compactor
    """
    
    @classmethod
    def compact_slots(
        cls,
        compacted_indices_dict,
        dst_slot_mappings,
        seq_group: SequenceGroup):
        """
        Perform slot/metadata compaction in scheduler.
        Update dst_slot_mapping
        
        """
        for seq in seq_group.get_seqs():
            seq_id = seq.seq_id
            # Check whether the current seq_id needs to be compacted
            if seq_id not in compacted_indices_dict:
                continue
            
            # Get block tables
            # NOTE: block table object is under vllm.block
            # not in vllm.core
            block_table = self.block_manager.block_tables[seq.seq_id]
            org_block_table_dict = {seq.seq_id: block_table._block_ids}

            # Construct original slot mapping
            org_slot_mapping = []
            skip_leading_tokens = 0
            vllm_block_size = 16
            # `-1` ignore newly generated token for now
            seq_len = seq.get_len() - 1
            compute_slot_mapping(False, org_slot_mapping, seq_id, seq_len, 
                skip_leading_tokens, 0, vllm_block_size, org_block_table_dict)
            
                
            # Free old block tables
            self.block_manager._free_block_table(block_table)
            
            # Update _prompt_token_ids and _output_token_ids
            compacted_indices = compacted_indices_dict[seq_id]
            compacted_prompt_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [])
            compacted_output_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [])
            
            
            prompt_len = len(seq._prompt_token_ids)
            for i in compacted_indices:
                if i < prompt_len:
                    compacted_prompt_token_ids.append(seq.data._prompt_token_ids[i])
                else:
                    compacted_output_token_ids.append(seq.data._output_token_ids[i-prompt_len])
            
            seq.data.update_compacted_prompt_token_ids(compacted_prompt_token_ids)
            seq.data.update_compacted_output_token_ids(compacted_output_token_ids)
            seq.data._num_computed_tokens = len(compacted_indices)
                    
            
            # Allocate new block tables
            is_encoder_decoder = seq_group.is_encoder_decoder()
            block_table: BlockTable = \
                self.block_manager._allocate_sequence(seq,
                                            seq_group.num_seqs(),
                                            is_encoder_decoder)
            
            # re-attch last token after block table allocation
            # as vllm scheduler will append a slot to it
            compacted_output_token_ids.append(seq.data._output_token_ids[-1])
            seq.data.update_compacted_output_token_ids(compacted_output_token_ids)
            
            # Update block table
            self.block_manager.block_tables[seq.seq_id] = block_table
            
            compacted_block_table_dict = {seq.seq_id: block_table._block_ids}
            
            # Construct compacted slot mapping
            compacted_slot_mapping = []
            skip_leading_tokens = 0
            vllm_block_size = 16
            seq_len = seq.get_len() - 1
            compute_slot_mapping(False, compacted_slot_mapping, seq_id, seq_len, 
                skip_leading_tokens, 0, vllm_block_size, compacted_block_table_dict)
            
            
            # Update dst_slot_mapping
            dst_slot_mappings[seq_id] = compacted_slot_mapping
        