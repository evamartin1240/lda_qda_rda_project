#!/bin/bash

# Define the dataset path
DATASET="./data/complex_dataset.csv"

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset '$DATASET' not found."
    exit 1
fi

echo "Running LDA test..."
python experiments/test_lda.py "$DATASET" || exit 1

echo "Running QDA test..."
python experiments/test_qda.py "$DATASET" || exit 1

echo "Running RDA test..."
python experiments/test_rda.py "$DATASET" || exit 1

echo "Running RDA alpha sweep..."
python experiments/sweep_rda_alpha.py "$DATASET" || exit 1

echo "Running LDA vs QDA vs RDA comparison..."
python experiments/compare_lda_qda_rda.py "$DATASET" || exit 1

echo "All experiments completed successfully."
