from utils.common_functions import read_yaml
from config.paths_config import *
from src.data_ingestion import  DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining

if __name__=="__main__":
    data_ingestion = DataIngestion(config=read_yaml(CONFIG_PATH))
    data_ingestion.run()

    data_processor = DataProcessor(input_file=ANIMELIST_CSV, output_dir=PROCESSED_DIR)
    data_processor.run()

    model_trainer = ModelTraining(data_path=PROCESSED_DIR, config_path=CONFIG_PATH)
    model_trainer.train_model()