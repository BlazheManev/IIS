import os
import yaml
import numpy as np
import pandas as pd
from lxml import etree as ET


def preprocess_air_data():
    # Load params (if exist)
    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f).get("preprocess", {})
    except FileNotFoundError:
        params = {}

    station_filter = params.get("station", None)

    # Load XML
    with open("data/raw/air/air_data.xml", "rb") as file:
        tree = ET.parse(file)
        root = tree.getroot()

    print(f"Version: {root.attrib['verzija']}")
    print(f"Source: {root.find('vir').text}")
    print(f"Suggested Capture: {root.find('predlagan_zajem').text}")
    print(f"Suggested Capture Period: {root.find('predlagan_zajem_perioda').text}")
    print(f"Preparation Date: {root.find('datum_priprave').text}")

    sifre = sorted(set(elem.attrib["sifra"] for elem in root.findall(".//postaja")))

    if station_filter:
        if station_filter not in sifre:
            raise ValueError(f"Station '{station_filter}' not found in XML. Available: {sifre}")
        sifre = [station_filter]

    os.makedirs("data/preprocessed/air", exist_ok=True)

    for sifra in sifre:
        postaje = root.xpath(f'//postaja[@sifra="{sifra}"]')
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
        df = df.sort_values(by="date_to")

        path = f"data/preprocessed/air/{sifra}.csv"
        df.to_csv(path, index=False)
        print(f"✅ Saved: {path}")


if __name__ == "__main__":
    preprocess_air_data()
