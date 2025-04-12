import os
import numpy as np
import pandas as pd
from lxml import etree as ET


def preprocess_air_data():
    # Open XML file
    with open("data/raw/air/air_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    print(f"📄 Version: {root.attrib['verzija']}")
    print(f"📄 Source: {root.find('vir').text}")
    print(f"📄 Suggested Capture: {root.find('predlagan_zajem').text}")
    print(f"📄 Suggested Capture Period: {root.find('predlagan_zajem_perioda').text}")
    print(f"📄 Preparation Date: {root.find('datum_priprave').text}")

    os.makedirs("data/preprocessed/air", exist_ok=True)

    # Get all station codes
    station_codes = set(elem.attrib["sifra"] for elem in root.findall(".//postaja"))
    print(f"📡 Found stations: {station_codes}")

    for sifra in station_codes:
        postaje = root.findall(f".//postaja[@sifra='{sifra}']")
        rows = []

        for postaja in postaje:
            date_to = postaja.findtext("datum_do")
            pm10 = postaja.findtext("pm10") or np.nan
            pm2_5 = postaja.findtext("pm2.5") or np.nan
            rows.append([date_to, pm10, pm2_5])

        df = pd.DataFrame(rows, columns=["date_to", "PM10", "PM2.5"])
        df = df.replace("", np.nan)
        df = df.replace("<1", 1).replace("<2", 2)
        df["date_to"] = pd.to_datetime(df["date_to"])
        df["PM10"] = pd.to_numeric(df["PM10"], errors="coerce")
        df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
        df = df.drop_duplicates(subset=["date_to"])
        df = df.sort_values("date_to")

        output_path = f"data/preprocessed/air/{sifra}.csv"
        df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    preprocess_air_data()
