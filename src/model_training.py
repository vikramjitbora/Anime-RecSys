import os
import joblib
import numpy as np
import comet_ml
from dotenv import load_dotenv
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping
)

from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from utils.common_functions import read_yaml
from config.paths_config import *

load_dotenv()
logger = get_logger(__name__)


class ModelTraining:
    """
    Handles loading data, training the model, saving weights,
    and logging metrics/artifacts to Comet-ML.
    """

    def __init__(self, data_path, config_path):
        self.data_path = data_path
        self.config_path = config_path
        self.train_config = read_yaml(config_path)["training"]

        self.experiment = comet_ml.Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="anime-recsys",
            workspace="vikramjitbora"
        )
        logger.info("Model Training and Comet-ML initialized.")

    def load_data(self):
        """Loads training and testing data arrays."""
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)

            logger.info("Data loaded successfully for model training.")
            return X_train_array, X_test_array, y_train, y_test
        except Exception as e:
            raise CustomException("Failed to load data", e)

    def train_model(self):
        """
        Loads the data, builds and trains the model using parameters from config.
        Saves the model and logs training metrics to Comet-ML.
        """
        try:
            X_train_array, X_test_array, y_train, y_test = self.load_data()

            n_users = len(joblib.load(ENCODED_USER))
            n_anime = len(joblib.load(ENCODED_ANIME))

            base_model = BaseModel(config_path=self.config_path)
            model = base_model.RecommenderNet(n_users=n_users, n_anime=n_anime)

            # Get training params from config
            batch_size = self.train_config['batch_size']
            epochs = self.train_config['epochs']
            schedule = self.train_config['learning_rate_schedule']

            start_lr = schedule['start_lr']
            max_lr = schedule['max_lr']
            min_lr = schedule['min_lr']
            rampup_epochs = schedule['rampup_epochs']
            sustain_epochs = schedule['sustain_epochs']
            exp_decay = schedule['exp_decay']

            def lrfn(epoch):
                if epoch < rampup_epochs:
                    return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
                elif epoch < rampup_epochs + sustain_epochs:
                    return max_lr
                else:
                    return (max_lr - min_lr) * exp_decay ** (epoch - rampup_epochs - sustain_epochs) + min_lr

            lr_callback = LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=0)

            model_checkpoint = ModelCheckpoint(
                filepath=CHECKPOINT_FILE_PATH,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True
            )

            early_stopping = EarlyStopping(
                patience=3,
                monitor="val_loss",
                mode="min",
                restore_best_weights=True
            )

            callbacks = [model_checkpoint, lr_callback, early_stopping]

            os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH), exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            try:
                history = model.fit(
                    x=X_train_array,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test_array, y_test),
                    callbacks=callbacks
                )

                model.load_weights(CHECKPOINT_FILE_PATH)
                logger.info("Model training completed.")

                for epoch in range(len(history.history['loss'])):
                    self.experiment.log_metric('train_loss', history.history["loss"][epoch], step=epoch)
                    self.experiment.log_metric('val_loss', history.history["val_loss"][epoch], step=epoch)

            except Exception as e:
                raise CustomException("Model training failed.", e)

            self.save_model_weights(model)

        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during model training process", e)

    def extract_weights(self, layer_name, model):
        """Extracts and normalizes the embedding weights of a given model layer."""
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))

            logger.info(f"Extracting weights for {layer_name}")
            return weights
        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during weight extraction process", e)

    def save_model_weights(self, model):
        """Saves trained model and embeddings to disk, and logs them as Comet-ML artifacts."""
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")

            user_weights = self.extract_weights('user_embedding', model)
            anime_weights = self.extract_weights('anime_embedding', model)

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            self.experiment.log_asset(MODEL_PATH)
            self.experiment.log_asset(ANIME_WEIGHTS_PATH)
            self.experiment.log_asset(USER_WEIGHTS_PATH)

            logger.info("User and anime weights saved successfully.")
        except Exception as e:
            logger.error(str(e))
            raise CustomException("Error during saving model and weights process", e)


if __name__ == "__main__":
    model_trainer = ModelTraining(data_path=PROCESSED_DIR, config_path=CONFIG_PATH)
    model_trainer.train_model()
