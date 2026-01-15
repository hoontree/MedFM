#!/bin/bash
# Batch convert all TinyUSFM datasets to COCO format

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE="${PROJECT_ROOT}/data/coco_format"

echo "=========================================="
echo "Converting TinyUSFM Datasets to COCO Format"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Output Base: $OUTPUT_BASE"
echo ""

# Array of datasets to convert
DATASETS=(
    "BUSBRA"
    "BUSI"
    "BUID"
    "BUS_UCLM"
    "B"
)

# Convert each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Converting: $dataset"
    echo "=========================================="

    config_file="${PROJECT_ROOT}/config/data/${dataset}.yaml"
    output_dir="${OUTPUT_BASE}/${dataset}"

    if [ ! -f "$config_file" ]; then
        echo "Warning: Config file not found: $config_file"
        echo "Skipping $dataset"
        continue
    fi

    python3 "${SCRIPT_DIR}/convert_to_coco.py" \
        --config "$config_file" \
        --output_dir "$output_dir"

    if [ $? -eq 0 ]; then
        echo "✓ Successfully converted $dataset"
    else
        echo "✗ Failed to convert $dataset"
    fi
done

echo ""
echo "=========================================="
echo "Conversion Complete!"
echo "=========================================="
echo "All COCO format datasets saved to: $OUTPUT_BASE"
echo ""
echo "Directory structure:"
tree -L 2 "$OUTPUT_BASE" 2>/dev/null || find "$OUTPUT_BASE" -maxdepth 2 -type f -name "*.json"
