from ..config import BackendConfig
from .base import IntelligenceBackend, register_backend


# An Error class for the human backend
class HumanBackendError(Exception):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        super().__init__(f"Human backend requires a UI to get input from {agent_name}.")


@register_backend
class Human(IntelligenceBackend):
    stateful = False
    type_name = "human"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_config(self) -> BackendConfig:
        return BackendConfig(backend_type=self.type_name)

    def query(self, agent_name: str, **kwargs) -> str:
        return "Answer the question as accurately as possible based on the information given, and put the answer in the form [answer]. Here is an example: Question: who was the first person killed in a car accident? Answer: Let’s think step by step! This tragic event occurred on August 17, 1896, in London, England. Bridget Driscoll was struck and killed by an automobile driven by Arthur Edsall, who was driving at a speed of approximately 4 miles per hour (about 6.4 kilometers per hour) at the Crystal Palace in London. Therefore, the answer is [Bridget Driscoll]. (END OF EXAMPLE) Evidence: (0) On July 20, 1969, American astronauts Neil Armstrong (1930-2012) and Edwin ¨BuzzÄldrin (1930-) became the first humans ever to land on the moon. (1) Neil Armstrong on the Moon. At 02:56 GMT on 21 July 1969, Armstrong became the first person to step onto the Moon. He was joined by Aldrin 19 minutes (2) Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri Question: On what date in 1969 did Neil Armstrong first set foot on the Moon? Answer: Let’s think step by step!"
        # raise HumanBackendError(agent_name)
