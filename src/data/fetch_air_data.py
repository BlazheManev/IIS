import requests
from datetime import datetime
import os

import yaml


def fetch_air_data():
    # Load configuration from YAML file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    param_path = os.path.join(project_root, "params.yaml")

    params = yaml.safe_load(open(param_path))["fetch"]
    try:
        # URL to fetch the XML data
        url = params["url"]

        # Fetch the XML data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the XML data to a file
        file_path = "data/raw/air/air_data.xml"
        with open(file_path, "wb") as file:
            file.write(response.content)

        # Print success message with file location and datetime
        print(f"Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        # Print error message if there is a problem fetching the file
        print(f"Error fetching data: {e}")


if __name__ == "__main__":
    fetch_air_data()