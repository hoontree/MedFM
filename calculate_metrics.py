"""
Segmentation Metrics Calculator

Calculates comprehensive segmentation metrics from prediction PNG images and ground truth images.
Metrics include: Dice, IoU, HD95, Sensitivity, Specificity, Precision, Recall, F1, ASD, and more.

Supports uploading results to Notion database.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial.distance import directed_hausdorff
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Notion imports (optional)
try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False


def load_mask(path: str, threshold: int = 127) -> np.ndarray:
    """
    Load a mask image and convert to binary.

    Args:
        path: Path to the mask image
        threshold: Threshold for binarization (default: 127)

    Returns:
        Binary mask as numpy array (0 and 1) with int64 dtype to prevent overflow
    """
    img = Image.open(path).convert('L')
    mask = np.array(img)
    # Use int64 to prevent overflow in metric calculations (e.g., tp * tn for large images)
    binary_mask = (mask > threshold).astype(np.int64)
    return binary_mask


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Dice Similarity Coefficient (DSC).

    DSC = 2 * |pred ∩ gt| / (|pred| + |gt|)
    """
    intersection = np.sum(pred * gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    if pred_sum + gt_sum == 0:
        return 1.0  # Both empty, perfect match

    return (2.0 * intersection) / (pred_sum + gt_sum)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU / Jaccard Index).

    IoU = |pred ∩ gt| / |pred ∪ gt|
    """
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection

    if union == 0:
        return 1.0  # Both empty, perfect match

    return intersection / union


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Sensitivity (True Positive Rate / Recall).

    Sensitivity = TP / (TP + FN)
    """
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)

    if tp + fn == 0:
        return 1.0  # No positive samples in GT

    return tp / (tp + fn)


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)
    """
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred * (1 - gt))

    if tn + fp == 0:
        return 1.0  # No negative samples in GT

    return tn / (tn + fp)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Precision (Positive Predictive Value).

    Precision = TP / (TP + FP)
    """
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))

    if tp + fp == 0:
        return 1.0  # No positive predictions

    return tp / (tp + fp)


def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Pixel Accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    correct = np.sum(pred == gt)
    total = pred.size
    return correct / total


def f1_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate F1 Score (harmonic mean of precision and recall).

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    prec = precision(pred, gt)
    rec = sensitivity(pred, gt)  # Recall = Sensitivity

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract surface (boundary) points from a binary mask.
    """
    if np.sum(mask) == 0:
        return np.array([])

    # Surface is the boundary: mask - eroded mask
    eroded = binary_erosion(mask)
    surface = mask.astype(bool) & ~eroded
    return np.argwhere(surface)


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Hausdorff Distance (HD).

    HD = max(h(pred, gt), h(gt, pred))
    where h(A, B) = max_{a ∈ A} min_{b ∈ B} ||a - b||

    Returns NaN if either mask is empty (excluded from statistics).
    """
    pred_points = get_surface_points(pred)
    gt_points = get_surface_points(gt)

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.nan  # Use NaN to exclude from mean/std calculations

    hd1 = directed_hausdorff(pred_points, gt_points)[0]
    hd2 = directed_hausdorff(gt_points, pred_points)[0]

    return max(hd1, hd2)


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate 95th percentile Hausdorff Distance (HD95).

    More robust to outliers than standard HD.
    Returns NaN if either mask is empty (excluded from statistics).
    """
    pred_points = get_surface_points(pred)
    gt_points = get_surface_points(gt)

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.nan  # Use NaN to exclude from mean/std calculations

    # Calculate all pairwise distances from pred surface to gt surface
    from scipy.spatial.distance import cdist

    # Distances from each pred point to nearest gt point
    distances_pred_to_gt = cdist(pred_points, gt_points, 'euclidean').min(axis=1)
    # Distances from each gt point to nearest pred point
    distances_gt_to_pred = cdist(gt_points, pred_points, 'euclidean').min(axis=1)

    # Combine all distances and take 95th percentile
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])

    return np.percentile(all_distances, 95)


def average_surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Average Surface Distance (ASD).

    ASD = (mean(d(pred_surface, gt)) + mean(d(gt_surface, pred))) / 2
    Returns NaN if either mask is empty (excluded from statistics).
    """
    pred_points = get_surface_points(pred)
    gt_points = get_surface_points(gt)

    if len(pred_points) == 0 and len(gt_points) == 0:
        return 0.0
    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.nan  # Use NaN to exclude from mean/std calculations

    from scipy.spatial.distance import cdist

    # Mean distance from pred surface to gt surface
    distances_pred_to_gt = cdist(pred_points, gt_points, 'euclidean').min(axis=1)
    mean_pred_to_gt = np.mean(distances_pred_to_gt)

    # Mean distance from gt surface to pred surface
    distances_gt_to_pred = cdist(gt_points, pred_points, 'euclidean').min(axis=1)
    mean_gt_to_pred = np.mean(distances_gt_to_pred)

    return (mean_pred_to_gt + mean_gt_to_pred) / 2


def volume_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Volume Similarity (VS).

    VS = 1 - |V_pred - V_gt| / (V_pred + V_gt)
    """
    pred_vol = np.sum(pred)
    gt_vol = np.sum(gt)

    if pred_vol + gt_vol == 0:
        return 1.0

    return 1 - abs(pred_vol - gt_vol) / (pred_vol + gt_vol)


def relative_volume_difference(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Relative Volume Difference (RVD).

    RVD = (V_pred - V_gt) / V_gt
    Positive: over-segmentation, Negative: under-segmentation
    """
    pred_vol = np.sum(pred)
    gt_vol = np.sum(gt)

    if gt_vol == 0:
        if pred_vol == 0:
            return 0.0
        return np.inf

    return (pred_vol - gt_vol) / gt_vol


def matthews_correlation_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC).

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    """
    # Use float64 to prevent overflow when multiplying large pixel counts
    tp = float(np.sum(pred * gt))
    tn = float(np.sum((1 - pred) * (1 - gt)))
    fp = float(np.sum(pred * (1 - gt)))
    fn = float(np.sum((1 - pred) * gt))

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def cohen_kappa(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Cohen's Kappa coefficient.

    Measures agreement between prediction and ground truth,
    accounting for chance agreement.
    """
    # Use float64 to prevent overflow when multiplying large pixel counts
    tp = float(np.sum(pred * gt))
    tn = float(np.sum((1 - pred) * (1 - gt)))
    fp = float(np.sum(pred * (1 - gt)))
    fn = float(np.sum((1 - pred) * gt))

    total = tp + tn + fp + fn
    po = (tp + tn) / total  # Observed agreement

    # Expected agreement by chance
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total ** 2)

    if 1 - pe == 0:
        return 1.0 if po == 1.0 else 0.0

    return (po - pe) / (1 - pe)


def balanced_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate Balanced Accuracy.

    BA = (Sensitivity + Specificity) / 2
    """
    sens = sensitivity(pred, gt)
    spec = specificity(pred, gt)
    return (sens + spec) / 2


def false_positive_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate False Positive Rate (FPR).

    FPR = FP / (FP + TN) = 1 - Specificity
    """
    return 1 - specificity(pred, gt)


def false_negative_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Calculate False Negative Rate (FNR).

    FNR = FN / (FN + TP) = 1 - Sensitivity
    """
    return 1 - sensitivity(pred, gt)


def calculate_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Calculate all segmentation metrics.

    Args:
        pred: Binary prediction mask
        gt: Binary ground truth mask

    Returns:
        Dictionary containing all metric values
    """
    metrics = {
        'Dice': dice_coefficient(pred, gt),
        'IoU': iou_score(pred, gt),
        'Sensitivity': sensitivity(pred, gt),
        'Specificity': specificity(pred, gt),
        'Precision': precision(pred, gt),
        'Recall': sensitivity(pred, gt),  # Same as sensitivity
        'F1': f1_score(pred, gt),
        'Accuracy': accuracy(pred, gt),
        'Balanced_Accuracy': balanced_accuracy(pred, gt),
        'HD': hausdorff_distance(pred, gt),
        'HD95': hausdorff_distance_95(pred, gt),
        'ASD': average_surface_distance(pred, gt),
        'Volume_Similarity': volume_similarity(pred, gt),
        'Relative_Volume_Diff': relative_volume_difference(pred, gt),
        'MCC': matthews_correlation_coefficient(pred, gt),
        'Cohen_Kappa': cohen_kappa(pred, gt),
        'FPR': false_positive_rate(pred, gt),
        'FNR': false_negative_rate(pred, gt),
    }

    return metrics


def find_matching_files(pred_dir: str, gt_dir: str,
                        pred_suffix: str = '', gt_suffix: str = '') -> List[Tuple[str, str]]:
    """
    Find matching prediction and ground truth file pairs.

    Args:
        pred_dir: Directory containing prediction masks
        gt_dir: Directory containing ground truth masks
        pred_suffix: Suffix to remove from prediction filenames for matching
        gt_suffix: Suffix to remove from GT filenames for matching

    Returns:
        List of (pred_path, gt_path) tuples
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    # Get all PNG files
    pred_files = list(pred_dir.glob('*.png'))
    gt_files = {f.stem.replace(gt_suffix, ''): f for f in gt_dir.glob('*.png')}

    pairs = []
    for pred_file in pred_files:
        # Get base name without suffix
        base_name = pred_file.stem.replace(pred_suffix, '')

        if base_name in gt_files:
            pairs.append((str(pred_file), str(gt_files[base_name])))

    return pairs


def calculate_metrics_for_directory(pred_dir: str, gt_dir: str,
                                    pred_suffix: str = '', gt_suffix: str = '',
                                    threshold: int = 127,
                                    output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate metrics for all matching images in directories.

    Args:
        pred_dir: Directory containing prediction masks
        gt_dir: Directory containing ground truth masks
        pred_suffix: Suffix to remove from prediction filenames for matching
        gt_suffix: Suffix to remove from GT filenames for matching
        threshold: Threshold for binarization
        output_csv: Path to save results CSV (optional)

    Returns:
        DataFrame with metrics for each image and summary statistics
    """
    pairs = find_matching_files(pred_dir, gt_dir, pred_suffix, gt_suffix)

    if len(pairs) == 0:
        print(f"Warning: No matching files found between {pred_dir} and {gt_dir}")
        return pd.DataFrame()

    print(f"Found {len(pairs)} matching image pairs")

    results = []
    for pred_path, gt_path in tqdm(pairs, desc="Calculating metrics"):
        pred_mask = load_mask(pred_path, threshold)
        gt_mask = load_mask(gt_path, threshold)

        metrics = calculate_all_metrics(pred_mask, gt_mask)
        metrics['filename'] = Path(pred_path).name
        results.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns to put filename first
    cols = ['filename'] + [c for c in df.columns if c != 'filename']
    df = df[cols]

    # Calculate summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary = df[numeric_cols].agg(['mean', 'std', 'min', 'max', 'median'])

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Print key metrics
    key_metrics = ['Dice', 'IoU', 'HD95', 'Sensitivity', 'Specificity', 'Precision', 'F1']
    for metric in key_metrics:
        if metric in summary.columns:
            print(f"\n{metric}:")
            print(f"  Mean:   {summary.loc['mean', metric]:.4f}")
            print(f"  Std:    {summary.loc['std', metric]:.4f}")
            print(f"  Median: {summary.loc['median', metric]:.4f}")
            print(f"  Min:    {summary.loc['min', metric]:.4f}")
            print(f"  Max:    {summary.loc['max', metric]:.4f}")

    # Save to CSV if specified
    if output_csv:
        df.to_csv(output_csv, index=False)

        # Also save summary
        summary_path = output_csv.replace('.csv', '_summary.csv')
        summary.to_csv(summary_path)

        print(f"\nResults saved to: {output_csv}")
        print(f"Summary saved to: {summary_path}")

    return df, summary


def get_notion_client(notion_token: str) -> "NotionClient":
    """Create a Notion client instance."""
    return NotionClient(auth=notion_token)


def get_existing_notion_properties(client: "NotionClient", database_id: str) -> Dict[str, str]:
    """
    Get existing property names and types from a Notion database.

    Args:
        client: Notion client instance
        database_id: Notion database ID

    Returns:
        Dictionary of property names to their types
    """
    db = client.databases.retrieve(database_id=database_id)
    return {name: prop["type"] for name, prop in db.get("properties", {}).items()}


def add_notion_properties(client: "NotionClient", database_id: str,
                          number_props: List[str], text_props: List[str] = None):
    """
    Add new properties to a Notion database.

    Args:
        client: Notion client instance
        database_id: Notion database ID
        number_props: List of number property names to add
        text_props: List of rich_text property names to add
    """
    properties = {}

    for prop_name in number_props:
        properties[prop_name] = {"number": {}}
        print(f"  Adding number column: {prop_name}")

    if text_props:
        for prop_name in text_props:
            properties[prop_name] = {"rich_text": {}}
            print(f"  Adding text column: {prop_name}")

    if properties:
        client.databases.update(database_id=database_id, properties=properties)


def upload_to_notion(
    summary: pd.DataFrame,
    database_id: str,
    notion_token: str,
    experiment_name: str,
    extra_properties: Optional[Dict[str, str]] = None
) -> str:
    """
    Upload metric summary to Notion database.

    Args:
        summary: DataFrame with summary statistics (mean, std, etc.)
        database_id: Notion database ID
        notion_token: Notion integration token
        experiment_name: Name/title for this experiment entry
        extra_properties: Additional text properties to add (e.g., {"Model": "UNet", "Dataset": "BUS"})

    Returns:
        Page ID of created Notion page
    """
    if not NOTION_AVAILABLE:
        raise ImportError("notion-client package not installed. Run: pip install notion-client")

    # Clean database_id (remove query params like ?v=... from Notion URL)
    database_id = database_id.split('?')[0].replace('-', '')

    # Create client
    client = get_notion_client(notion_token)

    # Get existing properties in the database
    existing_props = get_existing_notion_properties(client, database_id)
    print(f"Existing Notion columns: {len(existing_props)}")

    # Prepare metric properties: Metric_mean, Metric_std format
    metric_properties = {}
    stats_to_upload = ['mean', 'std']  # Upload mean and std

    for metric in summary.columns:
        for stat in stats_to_upload:
            if stat in summary.index:
                value = summary.loc[stat, metric]
                # Skip NaN and inf values
                if pd.isna(value) or np.isinf(value):
                    continue
                prop_name = f"{metric}_{stat}"
                metric_properties[prop_name] = float(value)

    # Check and add missing properties to database
    number_props_to_add = [p for p in metric_properties.keys() if p not in existing_props]
    text_props_to_add = []
    if extra_properties:
        text_props_to_add = [p for p in extra_properties.keys() if p not in existing_props]

    if number_props_to_add or text_props_to_add:
        print(f"Adding {len(number_props_to_add)} number columns and {len(text_props_to_add)} text columns...")
        add_notion_properties(client, database_id, number_props_to_add, text_props_to_add)
        # Refresh existing_props after adding new columns
        existing_props = get_existing_notion_properties(client, database_id)

    # Build page properties
    # Find title property name (it's usually "Name" or "Title")
    title_prop_name = None
    for prop_name, prop_type in existing_props.items():
        if prop_type == "title":
            title_prop_name = prop_name
            break

    if not title_prop_name:
        raise ValueError("No title property found in Notion database. Please create a title column first.")

    properties = {
        title_prop_name: {
            "title": [{"text": {"content": experiment_name}}]
        }
    }

    # Add extra text properties
    if extra_properties:
        for prop_name, prop_value in extra_properties.items():
            properties[prop_name] = {
                "rich_text": [{"text": {"content": prop_value}}]
            }

    # Add metric values
    for prop_name, value in metric_properties.items():
        properties[prop_name] = {
            "number": round(value, 6)
        }

    # Create the page
    result = client.pages.create(
        parent={"database_id": database_id},
        properties=properties
    )

    page_id = result.get("id", "unknown")
    print(f"\nUploaded to Notion! Page ID: {page_id}")

    return page_id


def main():
    parser = argparse.ArgumentParser(
        description='Calculate segmentation metrics from prediction and ground truth PNG images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python calculate_metrics.py --pred_dir ./predictions --gt_dir ./ground_truth
  python calculate_metrics.py --pred_dir ./preds --gt_dir ./gts --output results.csv
  python calculate_metrics.py --pred_dir ./preds --gt_dir ./gts --pred_suffix _pred --gt_suffix _gt

Notion upload:
  python calculate_metrics.py --pred_dir ./preds --gt_dir ./gts \\
      --notion_db DATABASE_ID --notion_token TOKEN --experiment_name "UNet_BUS"

  Or use environment variables:
  export NOTION_DATABASE_ID=your_database_id
  export NOTION_TOKEN=your_token
  python calculate_metrics.py --pred_dir ./preds --gt_dir ./gts --experiment_name "UNet_BUS"

Metrics calculated:
  - Dice (DSC): Dice Similarity Coefficient
  - IoU: Intersection over Union (Jaccard Index)
  - HD: Hausdorff Distance
  - HD95: 95th percentile Hausdorff Distance
  - ASD: Average Surface Distance
  - Sensitivity: True Positive Rate / Recall
  - Specificity: True Negative Rate
  - Precision: Positive Predictive Value
  - F1: F1 Score
  - Accuracy: Pixel Accuracy
  - Balanced Accuracy: (Sensitivity + Specificity) / 2
  - Volume Similarity: Volume-based similarity
  - Relative Volume Difference: Volume over/under-segmentation
  - MCC: Matthews Correlation Coefficient
  - Cohen's Kappa: Agreement coefficient
  - FPR: False Positive Rate
  - FNR: False Negative Rate
        """
    )

    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing prediction PNG masks')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth PNG masks')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (optional)')
    parser.add_argument('--pred_suffix', type=str, default='',
                        help='Suffix to remove from prediction filenames for matching')
    parser.add_argument('--gt_suffix', type=str, default='',
                        help='Suffix to remove from GT filenames for matching')
    parser.add_argument('--threshold', type=int, default=127,
                        help='Threshold for binarization (default: 127)')

    # Notion arguments
    parser.add_argument('--notion_db', type=str, default=None,
                        help='Notion database ID (or set NOTION_DATABASE_ID env var)')
    parser.add_argument('--notion_token', type=str, default=None,
                        help='Notion integration token (or set NOTION_TOKEN env var)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for Notion entry (required if uploading to Notion)')
    parser.add_argument('--notion_extras', type=str, nargs='*', default=None,
                        help='Extra properties for Notion in key=value format (e.g., Model=UNet Dataset=BUS)')

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.pred_dir):
        raise ValueError(f"Prediction directory not found: {args.pred_dir}")
    if not os.path.isdir(args.gt_dir):
        raise ValueError(f"Ground truth directory not found: {args.gt_dir}")

    # Calculate metrics
    df, summary = calculate_metrics_for_directory(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        pred_suffix=args.pred_suffix,
        gt_suffix=args.gt_suffix,
        threshold=args.threshold,
        output_csv=args.output
    )

    # Upload to Notion if requested
    notion_db = os.getenv('NOTION_DATABASE_ID') or args.notion_db
    notion_token = os.getenv('NOTION_TOKEN') or args.notion_token

    if notion_db and notion_token:
        if not args.experiment_name:
            # Use pred_dir name as default experiment name
            args.experiment_name = Path(args.pred_dir).name

        # Parse extra properties
        extra_props = None
        if args.notion_extras:
            extra_props = {}
            for item in args.notion_extras:
                if '=' in item:
                    key, value = item.split('=', 1)
                    extra_props[key] = value

        upload_to_notion(
            summary=summary,
            database_id=notion_db,
            notion_token=notion_token,
            experiment_name=args.experiment_name,
            extra_properties=extra_props
        )
    elif args.experiment_name:
        print("\nWarning: --experiment_name provided but Notion credentials not found.")
        print("Set --notion_db and --notion_token, or NOTION_DATABASE_ID and NOTION_TOKEN env vars.")

    return df


if __name__ == '__main__':
    main()
