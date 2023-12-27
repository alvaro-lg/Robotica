import tensorflow as tf

from pathlib import Path

from simulations.domain.factories.model_factory import ModelFactory
from simulations.infrastructure.model_repository import ModelRepository

MODEL_NAME = "model_ep87"
MODELS_PATH = Path("models")


class ModelTransformingService:

    @staticmethod
    def run():
        # Loading the model
        repo = ModelRepository(MODELS_PATH)
        model = repo.load(MODEL_NAME)

        # Transforming the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Storing the model
        repo.store_lite(tflite_model, MODEL_NAME)


if __name__ == '__main__':
    # Actual application startup
    ModelTransformingService.run()
