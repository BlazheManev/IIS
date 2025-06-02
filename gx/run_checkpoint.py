import os
import sys
import great_expectations as gx

context = gx.get_context()

datasource_name = "air_quality"
expectation_suite_name = "air_quality_suite"
base_path = "data"  # relative to project root

# Make sure the datasource is available
if datasource_name not in context.datasources:
    context.sources.add_pandas_filesystem(
        name=datasource_name,
        base_directory=base_path,
    )

datasource = context.get_datasource(datasource_name)

# Get all CSV files for stations
data_folder = os.path.join(base_path, "preprocessed", "air")
all_csvs = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

any_failed = False

for csv_file in all_csvs:
    station_id = os.path.splitext(csv_file)[0]  # E.g., E403
    asset_name = f"air_quality_{station_id.lower()}"  # unique asset name per station

    print(f"\n▶ Validating: {csv_file}")

    # Register the CSV file as an asset
    asset = datasource.add_csv_asset(
        name=asset_name,
        glob_directive=f"preprocessed/air/{csv_file}"
    )

    # Build batch request
    batch_request = asset.build_batch_request()

    # Create a unique checkpoint name
    checkpoint_name = f"checkpoint_{station_id.lower()}"

    # Add/update and run checkpoint
    checkpoint = context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[{
            "batch_request": batch_request,
            "expectation_suite_name": expectation_suite_name,
        }],
    )

    result = checkpoint.run()
    context.build_data_docs()

    if result["success"]:
        print(f"✅ {csv_file} PASSED validation")
    else:
        print(f"❌ {csv_file} FAILED validation")
        any_failed = True

# Final status
sys.exit(1 if any_failed else 0)
