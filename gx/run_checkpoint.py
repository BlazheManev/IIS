import os
import sys
import great_expectations as gx

# 1. Inicializacija konteksta
context = gx.get_context()

datasource_name = "air_quality"
base_dir = "../data/preprocessed/air"

# 2. Pridobi vse E*.csv datoteke
csv_files = [f for f in os.listdir(base_dir) if f.startswith("E") and f.endswith(".csv")]

# 3. Validiraj vsako datoteko posebej
all_passed = True

for csv_file in csv_files:
    station_code = csv_file.replace(".csv", "")
    asset_name = f"air_quality_data_{station_code}"
    checkpoint_name = f"checkpoint_{station_code}"

    print(f"\n🚦 Zagon checkpointa za: {station_code}")

    try:
        asset = context.get_datasource(datasource_name).get_asset(asset_name)
        checkpoint = context.get_checkpoint(checkpoint_name)
    except Exception as e:
        print(f"❌ Napaka pri nalaganju za {station_code}: {e}")
        all_passed = False
        continue

    # Zaženi checkpoint
    result = checkpoint.run(run_id=f"{station_code}_run")

    if result["success"]:
        print(f"✅ {station_code}: Validation passed!")
    else:
        print(f"❌ {station_code}: Validation failed!")
        all_passed = False

# 4. Zgradi data docs enkrat na koncu
context.build_data_docs()

# 5. Končna odločitev
if all_passed:
    print("\n✅ Vse validacije uspešne!")
    sys.exit(0)
else:
    print("\n❌ Ena ali več validacij je padla.")
    sys.exit(1)
