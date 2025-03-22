import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion class")
        try:
            # Use the correct path to the CSV file
            csv_path = os.path.join(project_root, 'notebook', 'StudentsPerformance.csv')
            df = pd.read_csv(csv_path)
            logging.info('Read the dataset as dataframe.')
            logging.info(f"DataFrame columns: {df.columns.tolist()}")

            # Create artifacts directory with absolute path
            artifacts_dir = os.path.join(project_root, 'artifacts')
            os.makedirs(artifacts_dir, exist_ok=True)
            
            train_path = os.path.join(artifacts_dir, "train.csv")
            test_path = os.path.join(artifacts_dir, "test.csv")

            logging.info("Train-test-split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)
            logging.info(f"Ingestion completed. Files saved at {artifacts_dir}")
            logging.info(f"Train set columns: {train_set.columns.tolist()}")
            logging.info(f"Test set columns: {test_set.columns.tolist()}")

            return (
                train_path,
                test_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    obj = DataIngestion(data_ingestion_config)
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
