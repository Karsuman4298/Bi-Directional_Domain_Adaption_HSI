#!/bin/bash

echo "=================================================================================="
echo " Extracting LaTeX Table Column for GatedAgentBiDA"
echo "=================================================================================="

# Run for Houston13 -> Houston18
python3 extract_latex_column.py --source_name Houston13 --target_name Houston18

# Run for Houston18 -> Houston13
python3 extract_latex_column.py --source_name Houston18 --target_name Houston13
