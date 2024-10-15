#!/usr/bin/env sh

set -eu

usage() {
  echo "Usage: sh prepare_dataset_split_for_simulation_and_poc.sh -w WORKSPACE_DIR [-h]"
  echo "Splits the dataset for simulation and proof-of-concept mode."
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

# Fetch and unzip the dataset
cd $workspace_dir/data
mkdir simulated_node_0
cd simulated_node_0
ln -s ../images/odelia_dataset_only_sub/00* ./
cd ..
mkdir simulated_node_1
cd simulated_node_1
ln -s ../images/odelia_dataset_only_sub/01* ./
cd ..

# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi

echo "Split created successfully."
