import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys

logger = get_logger(__name__)

class DataProcessor:
    """
    Class to process anime rating data, including loading, filtering, scaling,
    encoding, splitting, and saving data for further use in machine learning models.
    """
    
    def __init__(self, input_file, output_dir):
        """
        Initializes the DataProcessor with input file and output directory.

        Args:
            input_file (str): Path to the input CSV file.
            output_dir (str): Directory where processed data and artifacts will be stored.
        """
        self.input_file = input_file
        self.output_dir = output_dir

        # Data placeholders
        self.rating_df = None
        self.anime_df = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        # Encoders/Decoders
        self.encoded_user = {}
        self.decoded_user = {}
        self.encoded_anime = {}
        self.decoded_anime = {}

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Data Processor Initialized")

    def load_data(self, usecols):
        """
        Loads rating data from CSV file with selected columns.

        Args:
            usecols (list): List of columns to read from the CSV file.
        """
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=usecols)
            logger.info("Data loaded successfully")
        except Exception as e:
            raise CustomException("Failed to load data", sys)

    def filter_users(self, min_rating=400):
        """
        Filters out users who have rated fewer items than `min_rating`.

        Args:
            min_rating (int): Minimum number of ratings a user must have to be kept.
        """
        try:
            n_ratings = self.rating_df["user_id"].value_counts()
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(n_ratings[n_ratings >= min_rating].index)].copy()
            logger.info("Filtered users successfully")
        except Exception as e:
            raise CustomException("Failed to filter data", sys)

    def scale_ratings(self):
        """
        Scales rating values between 0 and 1 using min-max normalization.
        """
        try:
            min_rating = self.rating_df["rating"].min()
            max_rating = self.rating_df["rating"].max()
            self.rating_df["rating"] = self.rating_df["rating"].apply(
                lambda x: (x - min_rating) / (max_rating - min_rating)
            ).astype(np.float64)
            logger.info("Rating scaling completed")
        except Exception as e:
            raise CustomException("Failed to scale data", sys)

    def encode_data(self):
        """
        Encodes user IDs and anime IDs into numerical indices.
        """
        try:
            # Encode users
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.encoded_user = {x: i for i, x in enumerate(user_ids)}
            self.decoded_user = {i: x for i, x in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.encoded_user)

            # Encode anime
            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.encoded_anime = {x: i for i, x in enumerate(anime_ids)}
            self.decoded_anime = {i: x for i, x in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.encoded_anime)

            logger.info("User and anime encoding completed")
        except Exception as e:
            raise CustomException("Failed to encode data", sys)
    
    def split_data(self, test_size=1000, random_state=42):
        """
        Splits the dataset into training and testing sets using sklearn's train_test_split.

        Args:
            test_size (int or float): If int, number of test samples; if float, proportion of test samples.
            random_state (int): Random seed for reproducibility.
        """
        try:
            X = self.rating_df[["user", "anime"]].values
            y = self.rating_df["rating"].values

            # If test_size is an int, convert it to a proportion
            if isinstance(test_size, int):
                test_size = test_size / len(self.rating_df)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=True
            )

            self.X_train_array = [X_train[:, 0], X_train[:, 1]]
            self.X_test_array = [X_test[:, 0], X_test[:, 1]]
            self.y_train = y_train
            self.y_test = y_test

            logger.info("Data split into training and testing sets successfully using train_test_split")

        except Exception as e:
            raise CustomException("Failed to split data using train_test_split", sys)


    # def split_data(self, test_size=1000, random_state=42):
    #     """
    #     Splits the dataset into training and testing sets.

    #     Args:
    #         test_size (int): Number of test samples.
    #         random_state (int): Random seed for reproducibility.
    #     """
    #     try:
    #         self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    #         X = self.rating_df[["user", "anime"]].values
    #         y = self.rating_df["rating"]

    #         train_indices = self.rating_df.shape[0] - test_size
    #         X_train, X_test, y_train, y_test = (
    #             X[:train_indices],
    #             X[train_indices:],
    #             y[:train_indices],
    #             y[train_indices:]
    #         )

    #         self.X_train_array = [X_train[:, 0], X_train[:, 1]]
    #         self.X_test_array = [X_test[:, 0], X_test[:, 1]]
    #         self.y_train = y_train
    #         self.y_test = y_test

    #         logger.info("Data split into training and testing sets successfully")
    #     except Exception as e:
    #         raise CustomException("Failed to split data", sys)

    def save_artifacts(self):
        """
        Saves encoded data and processed datasets to files for future use.
        """
        try:
            artifacts = {
                "encoded_user": self.encoded_user,
                "decoded_user": self.decoded_user,
                "encoded_anime": self.encoded_anime,
                "decoded_anime": self.decoded_anime,
            }

            for name, data in artifacts.items():
                joblib.dump(data, os.path.join(self.output_dir, f"{name}.pkl"))
                logger.info(f"{name} saved successfully")

            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)
            self.rating_df.to_csv(RATING_DF, index=False)

            logger.info("All training/testing data and rating dataframe saved")
        except Exception as e:
            raise CustomException("Failed to save artifacts", sys)

    def process_anime_data(self):
        """
        Processes anime metadata and synopsis data and saves them to CSV files.
        """
        try:
            df = pd.read_csv(ANIME_CSV)
            cols = ["MAL_ID", "Name", "Genres", "sypnopsis"]
            synopsis_df = pd.read_csv(ANIMESYNOPSIS_CSV, usecols=cols)

            df = df.replace("Unknown", np.nan)
            df['eng_version'] = df['English name'].fillna(df['Name'])
            df.rename({'MAL_ID': 'anime_id'}, axis=1, inplace=True)

            df.sort_values(by=["Score"], inplace=True, ascending=False, kind="quicksort", na_position="last")

            df = df[["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Premiered", "Members"]]
            df.to_csv(ANIME_DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)

            logger.info("Anime and synopsis data processed and saved")
        except Exception as e:
            raise CustomException("Failed to process anime data", sys)

    def run(self):
        """
        Executes the full data processing pipeline.
        """
        try:
            self.load_data(usecols=["user_id", "anime_id", "rating"])
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()
            logger.info("Data processing pipeline completed successfully")
        except CustomException as e:
            logger.error(str(e))


if __name__ == "__main__":
    data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
    data_processor.run()
