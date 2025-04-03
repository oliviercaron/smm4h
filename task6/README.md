# Classification Script (`classify.py`)

This script performs text classification using trained Hugging Face Transformer models. It supports inference using a single model checkpoint or ensembling predictions from multiple models trained during cross-validation. It can also evaluate performance and show classification errors if ground truth labels are provided.

## Features

*   Classify text data from a CSV file.
*   Supports single model inference from a checkpoint folder.
*   Supports ensemble inference by averaging predictions from models in cross-validation fold directories.
*   Auto-detects single vs. ensemble mode based on the provided path.
*   Loads optimal classification thresholds stored in `threshold.txt` files.
*   Generates evaluation reports (Precision, Recall, F1, Confusion Matrix) if input data contains a `labels` column.
*   Displays sample classification errors (False Positives/Negatives) during evaluation.
*   Creates a prediction CSV and a submission ZIP file.

## Requirements

*   Python 3.7+
*   PyTorch
*   Transformers (`pip install transformers`)
*   Pandas (`pip install pandas`)
*   Scikit-learn (`pip install scikit-learn`)
*   Rich (`pip install rich`)
*   A trained model checkpoint or a set of CV fold results.

## Input Data

*   Place your input CSV file inside the `data/` directory.
*   The CSV **must** contain columns named `id` and `text`.
*   For evaluation and error analysis, the CSV **must also** contain a column named `labels` with integer values (e.g., 0 or 1).

## Model Input

*   **Single Model:** Provide the path to the directory containing the model checkpoint files (e.g., `pytorch_model.bin`, `config.json`). This directory should ideally also contain a `threshold.txt` file with the optimal threshold.
    *   Example structure: `./models/my_final_model/`
*   **Ensemble Mode:** Provide the path to the base directory containing the cross-validation fold results. The script expects subdirectories named like `fold_0`, `fold_1`, etc., within this base directory. Each fold directory should contain a model checkpoint (or `final_model` directory) and a corresponding `threshold.txt` file.
    *   Example structure: `./results/my_cv_run/fold_0/`, `./results/my_cv_run/fold_1/`, etc.

## Usage
    
    ```bash
    python classify.py --input_path <path_to_input_csv> --model_path <path_to_model_or_cv_base_dir> [--output_dir <output_directory>] [--ensemble] [--no_eval] [--no_error_analysis]
    ```

### Arguments

*   **`model_path_or_cv_dir`**: (Required) Path to either:
    *   A single model checkpoint directory (e.g., `./models/final_best_model`).
    *   The base directory containing CV fold results (e.g., `./results/my_cv_results`).
*   **`input_filename`**: (Required) The name of the CSV file inside the `data/` folder (e.g., `valid.csv`, `test_data.csv`).

### Common Options

*   `--final_threshold <float>`: Manually set a final classification threshold (overrides loaded/optimized thresholds).
*   `--batch_size <int>`: Set the inference batch size (default: 32).
*   `--max_length <int>`: Override the maximum sequence length for tokenization.
*   `--base_model_hub_name <name>`: Specify the base model name on Hugging Face Hub (e.g., `bert-base-uncased`) if the tokenizer isn't found locally in the model path.
*   `--show_errors_count <int>`: Number of FP/FN examples to show during evaluation (default: 5).
*   `--do_ensemble`: Force ensemble mode even if detection fails (rarely needed).
    
### Examples

```bash
# Single model inference on validation data
python classify.py ./models/my_single_model data/valid.csv
```

# Ensemble inference on test data from CV results
python classify.py ./results/biomed_cv_corrected data/test_data.csv

# Single model inference, overriding threshold and showing 10 errors
python classify.py ./models/my_single_model data/valid.csv --final_threshold 0.65 --show_errors_count 10