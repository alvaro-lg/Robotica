from pathlib import Path

from simulations.domain.factories.model_factory import ModelFactory
from simulations.infrastructure.model_repository import ModelRepository

MODEL_NAME = "model"
MODELS_PATH = Path("models")


class ModelBuildingService:

    @staticmethod
    def run():
        repo = ModelRepository(MODELS_PATH)
        repo.store(ModelFactory.new(), MODEL_NAME)


if __name__ == '__main__':
    # Actual application startup
    ModelBuildingService.run()
