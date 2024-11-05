from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput

def new_gpu_executor_execute_model(
    self, execute_model_req: ExecuteModelRequest, compactor_input
) -> Optional[List[Union[SamplerOutput, PoolerOutput]]]:
    
    # Jiayi Modification starts
    output, compactor_output = self.driver_worker.execute_model(execute_model_req,
                                                                compactor_input)
    return output, compactor_output
    # Jiayi Modification ends