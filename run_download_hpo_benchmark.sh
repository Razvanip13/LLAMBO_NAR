#!/bin/bash

# Shell script to download HPObench Tabular Benchmark files

# URLs from the image
url_xgb="https://ndownloader.figshare.com/files/30469920"
url_svm="https://ndownloader.figshare.com/files/30379359"
url_lr="https://ndownloader.figshare.com/files/30379038"
url_rf="https://ndownloader.figshare.com/files/30469089"
url_nn="https://ndownloader.figshare.com/files/30379005"

# Define output filenames (assuming zip files based on 'Unzip' instruction)
file_xgb="hpo_xgb.zip"
file_svm="hpo_svm.zip"
file_lr="hpo_lr.zip"
file_rf="hpo_rf.zip"
file_nn="hpo_nn.zip"

# Function to download a single file with basic error check
download_file() {
  local url="$1"
  local output_filename="$2"
  local name="$3" # Just for display message

  echo "Downloading $name file..."
  # Use wget: -O specifies output filename, --quiet for less verbose output (optional)
  # Remove --quiet if you want to see wget's progress meter
  wget -O "$output_filename" "$url" # --quiet

  # Check if wget was successful (exit code 0)
  if [ $? -ne 0 ]; then
    echo "Error downloading $name from $url. Please check the URL and your connection."
    # Optional: exit the script if one download fails
    # exit 1
  else
    echo "$name downloaded successfully as $output_filename."
  fi
  echo "---" # Separator
}

# --- Main Download Section ---
echo "Starting HPObench file downloads..."
echo "---"

download_file "$url_xgb" "$file_xgb" "xgb"
download_file "$url_svm" "$file_svm" "svm"
download_file "$url_lr" "$file_lr" "lr"
download_file "$url_rf" "$file_rf" "rf"
download_file "$url_nn" "$file_nn" "nn"

echo "All downloads attempted."
echo ""
echo "--- Unzipping Information ---"
echo "The image suggests unzipping files into a structure like:"
echo "hpo_benchmarks/hpo-bench-{type}/{openml_task_id}/"
echo "Where {type} is nn, rf, xgb, etc., and {openml_task_id} is specific to your task."
echo ""
echo "Example (you need to create directories and replace YOUR_TASK_ID):"
echo "# task_id=\"YOUR_TASK_ID\""
echo "# base_dir=\"hpo_benchmarks\""
echo "# mkdir -p \"\$base_dir/hpo-bench-xgb/\$task_id\""
echo "# unzip \"$file_xgb\" -d \"\$base_dir/hpo-bench-xgb/\$task_id/\""
echo "# # ... repeat for svm, lr, rf, nn ..."
echo ""
echo "Please adapt the unzipping commands based on your specific needs."