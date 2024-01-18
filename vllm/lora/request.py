from dataclasses import dataclass


@dataclass
class LoRARequest:
    """
    Request for a LoRA adapter.

    Note that this class should be be used internally. For online
    serving, it is recommended to not allow users to use this class but
    instead provide another layer of abstraction to prevent users from
    accessing unauthorized LoRA adapters.

    lora_id and lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.
    """

    lora_id: str
    lora_int_id: int
    lora_local_path: str

    def __eq__(self, value: object) -> bool:
        return isinstance(value, LoRARequest) and self.lora_id == value.lora_id

    def __hash__(self) -> int:
        return self.lora_int_id
