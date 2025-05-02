import os
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    """
    Reads and parses a YAML configuration file.

    Args:
        file_path (str): The full path to the YAML file.

    Returns:
        dict: Parsed contents of the YAML file as a dictionary.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at the specified path: {file_path}")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the YAML file: %s", file_path)
            return config

    except Exception as e:
        logger.error("Error while reading YAML file: %s", file_path)
        raise CustomException("Failed to read YAML file", e)

    
    