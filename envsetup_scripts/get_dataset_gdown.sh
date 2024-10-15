#!/usr/bin/env sh

set -eu

usage() {
  echo "Usage: sh get_dataset_gdown.sh -w WORKSPACE_DIR [-h]"
  echo "Fetches the dataset from google drive and unzips it into the workspace directory."
  echo ""
  echo "Options:"
  echo "  -w WORKSPACE_DIR     Workspace directory"
  echo "  -h                   Show this help message"
  exit 1
}

# Parse command-line options
while getopts ":w:h" opt; do
    case $opt in
        w)
            workspace_dir="$OPTARG"
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

if ! command -v gdown 2>&1 >/dev/null
then
    echo "gdown could not be found, install it using"
    echo "pip install -U --no-cache-dir gdown --pre"
    exit 1
fi

# Fetch and unzip the dataset
mkdir -p $workspace_dir/data
gdown --fuzzy https://drive.google.com/drive/folders/1clbK91sQv8bYtGAhC9AmZAbgVo473yrz?usp=share_link -O $workspace_dir/data --folder
gdown --fuzzy https://drive.google.com/file/d/1HKjqbALJgZCqLeNJ-_xKk0lEDNcJV_5Z/view?usp=share_link   -O $workspace_dir/data/odelia_dataset_only_sub.zip
unzip $workspace_dir/data/odelia_dataset_only_sub.zip -d $workspace_dir/data/

# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi

echo "Dataset fetched successfully."
