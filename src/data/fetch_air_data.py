import requests
from datetime import datetime
import yaml
import os

def fetch_air_data():
    try:
        # Load URL from params.yaml
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["fetch"]
        url = params["url"]

        # Fetch the XML data
        response = requests.get(url)
        response.raise_for_status()

        # Ensure directory exists
        os.makedirs("data/raw/air", exist_ok=True)

        # Save the XML data to a file
        file_path = "data/raw/air/air_data.xml"
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"✅ Fetching successful. Data saved to {file_path} at {datetime.now()}")

    except requests.RequestException as e:
        print(f"❌ Error fetching data: {e}")

if __name__ == "__main__":
    fetch_air_data()
