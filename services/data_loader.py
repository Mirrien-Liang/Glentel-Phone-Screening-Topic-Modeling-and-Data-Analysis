from zipfile import BadZipfile
import pandas as pd
import os
from utils.preprocessing import preprocess_data


def load_data(preprocess: bool, file_path: str) -> pd.DataFrame:
    # Check if the file exists.
    if not os.path.exists(file_path):
        raise FileNotFoundError("The data file was not found.")

    # Load the data and create a dataframe.
    try:
        if file_path.endswith(".csv"):
            raw_data = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            raw_data = pd.read_excel(file_path, engine="openpyxl")

    except BadZipfile:
        # Error reading the Excel file.
        raise BadZipfile("The Excel file is corrupted.")
    
    except Exception as e:
        # Unknown error.
        raise e

    return preprocess_data(raw_data) if preprocess else raw_data


# if __name__ == "__main__":
#     from config import RAW_DATA_FILE_PATH
#     Test: Print the first 5 rows of the (preprocessed) input dataframe
#     load_data(False, RAW_DATA_FILE_PATH).head(5)
