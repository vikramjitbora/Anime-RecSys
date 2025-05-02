import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from dotenv import load_dotenv
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)
load_dotenv()


class DataIngestion:
    """
    Handles data ingestion from a Google Cloud Storage (GCS) bucket.

    This class reads configuration from a YAML file and downloads specified CSV files
    from the configured GCS bucket to a local directory. It also ensures proper 
    authentication using credentials from environment variables.

    Attributes:
        config (dict): Configuration for data ingestion parsed from YAML.
        bucket_name (str): GCS bucket name to download files from.
        file_names (list): List of filenames to download from the bucket.
        credentials_path (str): Path to GCP service account credentials.
    """

    def __init__(self, config_path):
        """
        Initializes the DataIngestion class with configuration and GCP credentials.

        Args:
            config_path (str): Path to the YAML configuration file.

        Raises:
            EnvironmentError: If GOOGLE_APPLICATION_CREDENTIALS is not set in environment.
        """
        self.config = read_yaml(config_path)["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["bucket_file_names"]
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not self.credentials_path:
            raise EnvironmentError('GOOGLE_APPLICATION_CREDENTIALS is not set in environment.')

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Initialized DataIngestion with bucket: {self.bucket_name}")

    def download_csv_from_gcp(self):
        """
        Downloads configured CSV files from the GCS bucket to the local RAW_DIR directory.

        - If a file already exists locally, it will be skipped.
        - For 'animelist.csv', only the first 15 million rows are retained to manage size.

        Raises:
            CustomException: If the file download fails for any reason.
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                if os.path.exists(file_path):
                    logger.info(f"File already exists: {file_name}. Skipping download.")
                    continue

                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)

                if file_name == 'animelist.csv':
                    data = pd.read_csv(file_path, nrows=15000000)
                    data.to_csv(file_path, index=False)
                    logger.info(f"Large file: {file_name} downloaded (15M rows only).")
                else:
                    logger.info(f"File: {file_name} downloaded successfully.")

        except Exception as e:
            logger.error(f"Failed to download file from bucket {self.bucket_name}")
            raise CustomException("Data download failed", e)

    def run(self):
        """
        Executes the full data ingestion process.

        This includes:
        - Downloading files from the GCS bucket.
        - Logging success or failure.
        """
        try:
            logger.info("Starting Data Ingestion Process....")
            self.download_csv_from_gcp()
            logger.info("Data Ingestion Completed...")
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
        finally:
            logger.info("Data Ingestion DONE.")
            
if __name__ == "__main__":
    data_ingestion = DataIngestion(config_path=CONFIG_PATH)
    data_ingestion.run()
