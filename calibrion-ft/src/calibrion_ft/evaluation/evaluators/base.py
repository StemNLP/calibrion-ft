from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def name(self) -> str:
        """Unique name for this evaluator."""
        pass

    def required_inputs(self) -> list:
        """
        Return a list of required input keys.
        e.g., ['html'], or ['expected', 'generated']
        """
        return []

    
    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Run the evaluation and return a result dictionary."""
        pass


