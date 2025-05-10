from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dot, Flatten, Dense, Activation, BatchNormalization
)
from utils.common_functions import read_yaml
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)


class BaseModel:
    """
    BaseModel is responsible for creating and compiling a recommendation model
    based on collaborative filtering using embeddings and dot product.

    Attributes:
        config (dict): Configuration parameters loaded from a YAML file.
    """

    def __init__(self, config_path):
        """
        Initializes the BaseModel by loading configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        try:
            self.config = read_yaml(config_path)
            logger.info("Loaded configuration from config.yaml")
        except Exception as e:
            raise CustomException("Error loading configuration", e)

    def RecommenderNet(self, n_users, n_anime):
        """
        Builds and compiles a recommendation neural network model.

        This model uses embedding layers to represent users and anime,
        followed by a dot product to compute similarity, and a dense
        layer to produce the final output.

        Args:
            n_users (int): Total number of unique users.
            n_anime (int): Total number of unique anime items.

        Returns:
            keras.Model: A compiled Keras model ready for training.
        """
        try:
            embedding_size = self.config["model"]["embedding_size"]

            # Input layers
            user_input = Input(name="user", shape=(1,))
            anime_input = Input(name="anime", shape=(1,))

            # Embedding layers
            user_embedding = Embedding(
                name="user_embedding",
                input_dim=n_users,
                output_dim=embedding_size
            )(user_input)

            anime_embedding = Embedding(
                name="anime_embedding",
                input_dim=n_anime,
                output_dim=embedding_size
            )(anime_input)

            # Dot product
            dot_product = Dot(
                name="dot_product",
                normalize=False,
                axes=2
            )([user_embedding, anime_embedding])

            # Flatten and Dense layers
            x = Flatten()(dot_product)
            x = Dense(1, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)

            # Build and compile model
            model = Model(inputs=[user_input, anime_input], outputs=x)
            model.compile(
                loss=self.config["model"]["loss"],
                optimizer=self.config["model"]["optimizer"],
                metrics=self.config["model"]["metrics"]
            )

            logger.info("Model created successfully.")
            return model

        except Exception as e:
            logger.error(f"Error occurred during model architecture creation: {e}")
            raise CustomException("Failed to create model", sys)
