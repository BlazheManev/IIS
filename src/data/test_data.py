import sys
import os
import pandas as pd
from evidently import Report
from evidently.presets.dataset_stats import DataSummaryPreset
from evidently.presets.drift import DataDriftPreset

# Define paths
CURRENT_DIR = "data/preprocessed/air"
REFERENCE_DIR = "data/reference/air"
REPORT_DIR = "reports"

# Ensure report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)

# Track failed files
failed_files = []

# Process each CSV file in the current data directory
for filename in os.listdir(CURRENT_DIR):
    if not filename.endswith(".csv"):
        continue

    print(f"\n🔍 Testing file: {filename}")
    current_path = os.path.join(CURRENT_DIR, filename)
    reference_path = os.path.join(REFERENCE_DIR, filename)
    report_path = os.path.join(REPORT_DIR, f"{filename}_report.html")

    try:
        # Load current data
        current = pd.read_csv(current_path)

        # Load or initialize reference data
        if not os.path.exists(reference_path):
            print(f"🆕 Reference not found. Creating: {reference_path}")
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            current.to_csv(reference_path, index=False)

        reference = pd.read_csv(reference_path)

        # Remove known date columns
        for col in ["date_to", "timestamp", "date"]:
            if col in reference.columns:
                del reference[col]
            if col in current.columns:
                del current[col]

        # Drop columns that are entirely empty in either dataset
        empty_cols = [col for col in reference.columns if reference[col].isna().all() or current[col].isna().all()]
        if empty_cols:
            print(f"⚠️ Skipping empty columns: {empty_cols}")
            reference.drop(columns=empty_cols, inplace=True)
            current.drop(columns=empty_cols, inplace=True)

        # Skip if no columns left
        if reference.shape[1] == 0 or current.shape[1] == 0:
            raise ValueError("No valid columns left after dropping empty ones.")

        # Generate report
        report = Report(
            [DataSummaryPreset(), DataDriftPreset()],
            include_tests=True
        )
        result = report.run(reference_data=reference, current_data=current)

        # Save HTML report
        result.save_html(report_path)
        print(f"✅ Report saved: {report_path}")

        # Check test results
        all_tests_passed = True
        result_dict = result.dict()
        if "tests" in result_dict:
            for test in result_dict["tests"]:
                if test.get("status") != "SUCCESS":
                    all_tests_passed = False
                    break

        if all_tests_passed:
            print(f"✔️ {filename}: All tests passed.")
            os.remove(reference_path)
            current.to_csv(reference_path, index=False)
        else:
            print(f"❌ {filename}: Data drift detected.")

    except Exception as e:
        print(f"❌ {filename}: Failed with error: {e}")
        failed_files.append((filename, str(e)))

# Summary at end
if failed_files:
    print("\n⚠️ Some files failed:")
    for fname, err in failed_files:
        print(f" - {fname}: {err}")
else:
    print("\n🎉 All files processed successfully with no errors.")
