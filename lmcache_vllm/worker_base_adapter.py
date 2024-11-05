from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
import time
from vllm.distributed import broadcast_tensor_dict, get_pp_group, get_tp_group
import torch

def new_worker_base_execute_model(
    self,
    execute_model_req: Optional[ExecuteModelRequest] = None,
    compactor_input = None,
) -> Optional[List[SamplerOutput]]:
    """Executes at least one model step on the given sequences, unless no
    sequences are provided."""
    start_time = time.perf_counter()

    inputs = self.prepare_input(execute_model_req)
    if inputs is None:
        return None

    model_input, worker_input, kwargs = inputs
    num_steps = worker_input.num_steps

    self.execute_worker(worker_input)

    # If there is no input, we don't need to execute the model.
    if worker_input.num_seq_groups == 0:
        return []

    intermediate_tensors = None
    orig_model_execute_time = 0.0
    if not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group()))
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time):
            orig_model_execute_time = intermediate_tensors.tensors.get(
                "model_execute_time", torch.tensor(0)).item()

    # Jiayi Modification starts
    output, compactor_output = self.model_runner.execute_model(
        model_input=model_input,
        compactor_input=compactor_input,
        kv_caches=self.kv_cache[worker_input.virtual_engine]
        if self.kv_cache is not None else None,
        intermediate_tensors=intermediate_tensors,
        num_steps=num_steps,
        **kwargs,
    )
    # Jiayi Modification ends

    model_execute_time = time.perf_counter() - start_time
    if not get_pp_group().is_last_rank:
        # output is IntermediateTensors
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time):
            output.tensors["model_execute_time"] = torch.tensor(
                model_execute_time + orig_model_execute_time)
        get_pp_group().send_tensor_dict(output.tensors,
                                        all_gather_group=get_tp_group())
        return [None]
    if (self.observability_config is not None
            and self.observability_config.collect_model_execute_time
            and output is not None):
        for o in output:
            o.model_execute_time = (orig_model_execute_time +
                                    model_execute_time)
    
    # Jiayi Modification starts
    # output is List[SamplerOutput]
    return output, compactor_output
    # Jiayi Modification ends