#!/bin/bash

# Shell script to unzip downloaded HPObench files.
# Assumes the zip files contain the necessary subdirectory structure (including task IDs) internally.

# --- Configuration ---
# Base directory where hpo-bench-* directories will be created
output_base_dir="hpo_benchmarks"

# Array of model types
types=("xgb" "svm" "lr" "rf" "nn")

# Corresponding zip filenames (expected from the download script)
# Assumes files are in the current directory where the script is run
zip_files=("hpo_xgb.zip" "hpo_svm.zip" "hpo_lr.zip" "hpo_rf.zip" "hpo_nn.zip")

echo "Starting unzip process..."
echo "Output base directory: $output_base_dir"
echo "Assuming zip files contain the task ID directory structure internally (e.g., 146818/)."
echo "---"

# --- Main Unzipping Loop ---
for i in "${!types[@]}"; do
  type="${types[i]}"
  zip_file="${zip_files[i]}"

  # Define the target directory *parent* for this type.
  # We expect 'unzip' to create the task ID sub-directory within this parent.
  target_parent_dir="$output_base_dir/hpo-bench-$type"

  echo "Processing type: $type"

  # 1. Check if the source zip file exists
  if [ ! -f "$zip_file" ]; then
    echo "  Warning: Source file '$zip_file' not found in current directory. Skipping."
    echo "---"
    continue # Move to the next type
  fi
  echo "  Found source file: $zip_file"

  # 2. Create the target parent directory (e.g., hpo_benchmarks/hpo-bench-rf/)
  echo "  Ensuring target parent directory exists: $target_parent_dir"
  mkdir -p "$target_parent_dir"
  if [ $? -ne 0 ]; then
    echo "  Error: Could not create directory '$target_parent_dir'. Check permissions. Skipping."
    echo "---"
    continue
  fi

  # 3. Unzip the file into the target parent directory
  # The structure inside the zip (e.g., 146818/...) should be created automatically within target_parent_dir.
  echo "  Unzipping '$zip_file' into '$target_parent_dir/'..."
  # -o: overwrite files without prompting
  # -q: quiet mode (less verbose output)
  # -d: destination directory
  unzip -o -q "$zip_file" -d "$target_parent_dir"

  if [ $? -ne 0 ]; then
    echo "  Error: Failed to unzip '$zip_file'. It might be corrupted or permissions are wrong."
    # Consider adding 'exit 1' here if one failure should stop the whole process
  else
    echo "  Successfully unzipped '$zip_file'."
  fi
  echo "---" # Separator for clarity
done

echo "Unzipping process finished."
exit 0