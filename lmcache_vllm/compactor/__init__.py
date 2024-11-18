from lmcache_vllm.compactor.h2o_compactor import H2OCompactor
from lmcache_vllm.compactor.utils import CompactorInput, CompactorOutput
from lmcache_vllm.compactor.base_scheduler_compactor import BaseSchedulerCompactor

__all__ = ["H2OCompactor", "BaseSchedulerCompactor",
           "CompactorInput", "CompactorOutput"]

class LMCacheCompactorBuilder:
    _instances: Dict[str, BaseLocalCompactor] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        compactor_type = "H2O"
    ) -> BaseLocalCompactor:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        if instance_id not in cls._instances:
            if compactor_type == "H2O":
                compactor = H2OCompactor()
            cls._instances[instance_id] = compactor
            return compactor
        else:
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID, 
        or None if not found."""
        return cls._instances.get(instance_id)