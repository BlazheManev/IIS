import os
import sys
import great_expectations as gx

# 1. Initialize GE context
context = gx.get_context()

datasource_name = "air_quality"
base_dir = "../data/preprocessed/air"

# 2. Get all E*.csv files
csv_files = [f for f in os.listdir(base_dir) if f.startswith("E") and f.endswith(".csv")]

all_passed = True

for csv_file in csv_files:
    station_code = csv_file.replace(".csv", "")
    asset_name = f"air_quality_data_{station_code}"
    checkpoint_name = f"checkpoint_{station_code}"

    print(f"\n🚦 Zagon checkpointa za: {station_code}")

    # Ensure the asset is registered
    try:
        asset = context.get_datasource(datasource_name).get_asset(asset_name)
    except (LookupError, gx.exceptions.DataContextError):
        # Register asset if not found
        print(f"➕ Asset not found. Registering: {asset_name}")
        datasource = context.get_datasource(datasource_name)
        asset = datasource.add_csv_asset(
            name=asset_name,
            batching_regex=rf"{station_code}\.csv"
        )

    # Load checkpoint
    try:
        checkpoint = context.get_checkpoint(checkpoint_name)
    except gx.exceptions.CheckpointNotFoundError:
        print(f"❌ Checkpoint not found for {station_code}: {checkpoint_name}")
        all_passed = False
        continue

    # Run checkpoint
    try:
        result = checkpoint.run(run_id=f"{station_code}_run")
        if result["success"]:
            print(f"✅ {station_code}: Validation passed!")
        else:
            print(f"❌ {station_code}: Validation failed!")
            all_passed = False
    except Exception as e:
        print(f"❌ {station_code}: Checkpoint execution error: {e}")
        all_passed = False

# 4. Build data docs once at the end
context.build_data_docs()

# 5. Exit code based on success
if all_passed:
    print("\n✅ Vse validacije uspešne!")
    sys.exit(0)
else:
    print("\n❌ Ena ali več validacij je padla.")
    sys.exit(1)
