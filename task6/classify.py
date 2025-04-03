# Essential lightweight imports at the top
import os
import sys
import argparse
import glob
import zipfile
import numpy as np
import gc
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
# Heavier imports will be deferred into main()

# Setup Rich console
console = Console()

# --- TASK IDENTIFIER ---
# Just a string to identify the task number of SMM4H. This way I can copy and paste the script for other tasks without having to change the name of the script itself.
TASK_IDENTIFIER = "task6"
# --- END TASK IDENTIFIER ---

# ---------------------
# Utility Functions
# ---------------------
def load_threshold(model_path, default=0.5):
    """Loads the optimal threshold from a 'threshold.txt' file."""
    threshold_path = os.path.join(model_path, "threshold.txt")
    threshold = default
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path, "r") as f:
                threshold = float(f.read().strip())
        except Exception as e:
            console.print(f"[dim]Failed to read threshold file {threshold_path}: {e}. Using default {default}[/]", style="yellow")
            threshold = default
    # else: file doesn't exist, default is already set
    return threshold

# find_optimal_threshold and TextDataset need heavy imports, define inside main()
# show_errors helper function for error analysis (moved here from old script)
def show_errors(df, title, console, max_items=5):
    """Displays sample errors (FP/FN) using Rich Panel."""
    console.rule(f"[bold]{title}[/] ({len(df)} total)")
    if df.empty:
        console.print("[dim]None found.[/]")
        return
    # Use .iloc to avoid index issues if non-sequential after filtering/merging
    for i in range(min(max_items, len(df))):
        row = df.iloc[i]
        # Safely handle potential missing keys just in case merge had issues
        true_label = row.get('labels', 'N/A')
        pred_label = row.get('label', 'N/A')
        text = row.get('text', 'N/A')
        row_id = row.get('id', 'N/A')

        panel_content = (
            f"[bold cyan]ID:[/] {row_id}\n"
            f"[red]TRUE:[/] {true_label}  |  [green]PRED:[/] {pred_label}\n"
            f"[dim]Text:[/] {str(text)[:300]}..." # Limit text length
        )
        console.print(Panel(panel_content, expand=False, border_style="dim"))


# ---------------------
# Main function
# ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="Classify texts using a single model checkpoint OR an ensemble of models from a CV base directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("model_path_or_cv_dir", help="Path to EITHER a single model checkpoint folder OR the base directory containing CV fold results (e.g., './results_biomed_cv_corrected'). Auto-detects ensemble mode.")
    parser.add_argument("input_filename", help="Name of the input CSV file inside the 'data/' folder (e.g., 'valid.csv' or 'test_data.csv').")
    # Removed --cv_base_dir as it's redundant with the first argument in ensemble mode
    # parser.add_argument("--cv_base_dir", default="./results_biomed_cv_corrected", help="Base directory containing CV fold results (used if different from model_path_or_cv_dir).")
    parser.add_argument("--base_model_hub_name", default="allenai/biomed_roberta_base", help="Base model name from Hugging Face Hub for tokenizer fallback.")
    parser.add_argument("--do_ensemble", action="store_true", help="Force ensemble mode. If not set, mode is auto-detected based on presence of 'fold_*' subdirs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--max_length", type=int, default=None, help="Override max sequence length. If None, try to infer or use 512.")
    parser.add_argument("--final_threshold", type=float, default=None, help="Specify a final threshold for ensemble or single model. Overrides optimization/defaults.")
    parser.add_argument("--show_errors_count", type=int, default=5, help="Number of error samples (FP/FN) to display if evaluating.")

    args = parser.parse_args()

    # --- Defer heavier library imports ---
    console.print("[dim]Importing libraries...[/]")
    try:
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
        from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score
        from rich.progress import track
    except ImportError as e:
        console.print(f"[bold red]Error: Missing required library -> {e}[/]")
        sys.exit(1)
    console.print("[green]✓ Libraries imported.[/]")

    # --- Define classes/functions requiring heavy imports ---

    class TextDataset(Dataset):
        """PyTorch Dataset for text classification inference."""
        def __init__(self, texts, ids, tokenizer, max_length=512):
            self.texts = texts
            self.ids = ids
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx]) if self.texts[idx] is not None else ""
            id_ = self.ids[idx]
            try:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                return {"input_ids": encoding["input_ids"].squeeze(0),
                        "attention_mask": encoding["attention_mask"].squeeze(0),
                        "id": id_}
            except Exception as e:
                console.print(f"\n[bold red]Error tokenizing ID {id_}:[/] {e}")
                # Return dummy tensors BUT keep original ID to avoid length mismatches later
                return {"input_ids": torch.zeros(self.max_length, dtype=torch.long),
                        "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                        "id": id_}

    def find_optimal_threshold(labels, probs):
        """Finds the threshold maximizing the binary F1 score."""
        valid_indices = labels != -1 # Filter out labels marked as invalid (-1)
        labels = labels[valid_indices]
        probs = probs[valid_indices]
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            # console.print("[yellow]Not enough valid labels or classes to optimize threshold. Using 0.5.[/]")
            return 0.5, 0.0
        try:
            precision, recall, thresholds = precision_recall_curve(labels, probs, pos_label=1)
            # Handle cases where precision or recall might be zero, avoid division by zero
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
            f1_scores = np.nan_to_num(f1_scores) # Replace NaN with 0

            if len(f1_scores) == 0:
                # console.print("[yellow]No valid F1 scores calculated during threshold optimization. Using 0.5.[/]")
                return 0.5, 0.0

            best_f1_idx = np.argmax(f1_scores)
            best_f1 = f1_scores[best_f1_idx]
            # Ensure index is valid for thresholds array
            best_thresh = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1] if thresholds.size > 0 else 0.5

            # Optional: check F1 at default 0.5
            # preds_at_05 = (probs >= 0.5).astype(int); f1_at_05 = f1_score(labels, preds_at_05, pos_label=1, zero_division=0)
            # if f1_at_05 >= best_f1 * 0.99: return 0.5, f1_at_05
            return best_thresh, best_f1
        except Exception as e:
            console.print(f"[red]Error during threshold optimization: {e}. Using 0.5.[/]")
            return 0.5, 0.0

    # --- FIXED OUTPUT FILENAMES ---
    output_dir = "results"
    # Use the TASK_IDENTIFIER for specific output naming required by competitions
    output_csv_name = f"prediction_{TASK_IDENTIFIER}.csv" # <-- USE CONSTANT HERE
    output_csv_path = os.path.join(output_dir, output_csv_name)
    output_zip_name = "submission.zip"                   # <-- FIXED NAME REQUIRED
    output_zip_path = os.path.join(output_dir, output_zip_name)
    os.makedirs(output_dir, exist_ok=True)

    # --- Construct Input Path ---
    input_csv_path = os.path.join("data", args.input_filename)

    # --- Initial Checks & Mode Detection ---
    if not os.path.exists(input_csv_path):
        console.print(f"[bold red]❌ Input file not found:[/] {input_csv_path}")
        sys.exit(1)

    model_base_or_single_path = args.model_path_or_cv_dir
    # Auto-detect ensemble mode based on presence of fold_* subdirs, unless forced by flag
    potential_fold_dirs = glob.glob(os.path.join(model_base_or_single_path, "fold_*"))
    is_ensemble_mode = args.do_ensemble or (potential_fold_dirs and os.path.isdir(model_base_or_single_path))

    if is_ensemble_mode:
        mode_title = "Ensemble Mode Setup"
        cv_base_dir_path = model_base_or_single_path # In ensemble, the arg IS the base dir
        console.print(f"[cyan]Running in Ensemble Mode (Base Dir: '{cv_base_dir_path}').[/]")
        if not os.path.isdir(cv_base_dir_path):
             console.print(f"[bold red]❌ Ensemble base directory not found:[/] {cv_base_dir_path}")
             sys.exit(1)
    else:
        mode_title = "Single Model Mode Setup"
        single_model_path = model_base_or_single_path # In single mode, the arg IS the model path
        console.print(f"[cyan]Running in Single Model Mode (Model Path: '{single_model_path}').[/]")
        if not os.path.isdir(single_model_path):
             console.print(f"[bold red]❌ Specified single model path is not a valid directory:[/] {single_model_path}")
             sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [bold blue]{device}[/]")

    # --- Load Data ---
    console.print(Panel.fit(f"💾 Loading Data from {input_csv_path}", title="Data Loading", border_style="magenta"))
    try:
        df = pd.read_csv(input_csv_path)
        required_cols = ['id', 'text']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            console.print(f"[bold red]❌ Input CSV missing required columns: {missing}[/]")
            sys.exit(1)
        # Keep rows where essential columns are not null
        initial_rows = len(df)
        df.dropna(subset=required_cols, inplace=True)
        if len(df) < initial_rows:
             console.print(f"[yellow]Dropped {initial_rows - len(df)} rows due to missing 'id' or 'text'.[/]")
        if df.empty:
             console.print(f"[bold red]❌ No valid data remaining after dropping rows with missing ID/Text.[/]")
             sys.exit(1)
        console.print(f"[green]✓ Data loaded. Found {len(df)} rows.[/]")
    except Exception as e:
         console.print(f"[bold red]❌ Failed to load or process input CSV '{input_csv_path}': {e}[/]")
         sys.exit(1)

    # --- Store gold labels ---
    gold_labels_present = 'labels' in df.columns
    gold_map = None
    if gold_labels_present:
        console.print("Attempting to process 'labels' column for evaluation...")
        try:
            # Create a separate series for processing labels to avoid modifying df inplace unnecessarily
            labels_series = df['labels'].copy()
            original_count = len(labels_series.dropna())

            # Convert to numeric, coercing errors (like empty strings) to NaN
            labels_numeric = pd.to_numeric(labels_series, errors='coerce')
            valid_numeric_count = len(labels_numeric.dropna())
            num_coerced = original_count - valid_numeric_count
            if num_coerced > 0:
                 console.print(f"[yellow]  - Coerced {num_coerced} non-numeric label entries to NaN.[/]")

            # Create map only from valid numeric labels mapped to ID
            temp_df_for_map = pd.DataFrame({'id': df['id'], 'labels': labels_numeric})
            temp_df_for_map.dropna(subset=['labels'], inplace=True)
            temp_df_for_map['labels'] = temp_df_for_map['labels'].astype(int) # Convert valid ones to int

            gold_map = temp_df_for_map.set_index("id")["labels"].to_dict()

            if gold_map:
                console.print(f"[green]✓ Gold labels processed for {len(gold_map)} IDs.[/]")
                # Check if all rows had processable labels
                if len(gold_map) < len(df):
                     console.print(f"[yellow]  - Note: {len(df) - len(gold_map)} rows did not have a valid numeric label.[/]")
            else:
                 gold_labels_present = False # Set back to False if map is empty after processing
                 console.print(f"[yellow]⚠️ 'labels' column found, but no valid numeric labels could be processed.[/]")

        except Exception as e:
            gold_labels_present = False
            console.print(f"[yellow]⚠️ Error processing gold labels column: {e}. Evaluation will be skipped.[/]")
    else:
         console.print("[dim]No 'labels' column found in input data. Evaluation will be skipped.[/]")


    # --- Identify Model Paths & Load Individual Thresholds ---
    model_paths_to_load = []
    individual_thresholds = []
    tokenizer_load_path_candidate = None # Path from which to try loading the tokenizer

    if is_ensemble_mode:
        console.print(Panel.fit(f"🔍 Finding models for ensemble in {cv_base_dir_path}", title=mode_title, border_style="cyan"))
        fold_dirs = sorted(glob.glob(os.path.join(cv_base_dir_path, "fold_*")))
        if not fold_dirs:
            console.print(f"[bold red]❌ No 'fold_*' directories found in {cv_base_dir_path}. Cannot run ensemble.[/]")
            sys.exit(1)

        # Find the best checkpoint or final model within each fold dir
        for fold_dir in fold_dirs:
            best_ckpt_in_fold = None
            fold_threshold = 0.5
            # Prioritize 'final_model' if it exists and has a threshold file
            final_model_path = os.path.join(fold_dir, "final_model")
            if os.path.isdir(final_model_path) and os.path.exists(os.path.join(final_model_path, "threshold.txt")):
                best_ckpt_in_fold = final_model_path
                fold_threshold = load_threshold(best_ckpt_in_fold) # Load its specific threshold
            else:
                # If no final_model or it lacks threshold, look for best checkpoint-*
                potential_ckpts = sorted(glob.glob(os.path.join(fold_dir, "checkpoint-*")), reverse=True)
                for ckpt in potential_ckpts:
                    if os.path.isdir(ckpt) and os.path.exists(os.path.join(ckpt, "threshold.txt")):
                        best_ckpt_in_fold = ckpt
                        fold_threshold = load_threshold(best_ckpt_in_fold)
                        break # Found the best valid checkpoint

            if best_ckpt_in_fold:
                model_paths_to_load.append(best_ckpt_in_fold)
                individual_thresholds.append(fold_threshold)
                relative_path = os.path.relpath(best_ckpt_in_fold, cv_base_dir_path)
                console.print(f"  [green]✓[/] Using: [dim]{relative_path}[/] (Thr: {fold_threshold:.4f})")
            else:
                console.print(f"  [yellow]⚠️[/] No valid model (final_model or checkpoint-* with threshold.txt) found in [dim]{os.path.basename(fold_dir)}[/]. Skipping this fold.")

        if not model_paths_to_load:
            console.print(f"[bold red]❌ No valid models found across all folds in {cv_base_dir_path}. Aborting.[/]")
            sys.exit(1)

        console.print(f"[green]✓ Using {len(model_paths_to_load)} models for ensemble.[/]")
        tokenizer_load_path_candidate = model_paths_to_load[0] # Use first found model for tokenizer attempt
    else:
        # Single Model Mode
        console.print(Panel.fit(f"🔍 Setting up single model: {single_model_path}", title=mode_title, border_style="cyan"))
        # Check if threshold file exists for the single model
        threshold = load_threshold(single_model_path) # Load threshold specifically for this model
        if not os.path.exists(os.path.join(single_model_path, "threshold.txt")):
             console.print(f"[yellow]  - Note: threshold.txt not found in {single_model_path}. Using default 0.5 or --final_threshold if set.[/]")

        model_paths_to_load.append(single_model_path)
        individual_thresholds.append(threshold) # Store the loaded/default threshold
        tokenizer_load_path_candidate = single_model_path # Use this model path for tokenizer attempt
        console.print(f"  [green]✓[/] Using model: [dim]{os.path.basename(single_model_path)}[/] (Threshold found/default: {threshold:.4f})")

    # --- Load Tokenizer (once) ---
    console.print(Panel.fit("🔍 Loading Tokenizer (once)...", title="Setup", border_style="cyan"))
    tokenizer = None; tokenizer_source_info = "[red]Not loaded[/]"
    try:
        console.print(f"Attempting to load tokenizer from: [yellow]{tokenizer_load_path_candidate}[/]")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path_candidate)
        tokenizer_source_info = f"Local ([dim]{os.path.basename(tokenizer_load_path_candidate)}[/])"
        console.print(f"[green]✓ Tokenizer loaded successfully from local path.[/]")
    except Exception as e:
        console.print(f"[yellow]⚠️ Tokenizer not found or load error in {tokenizer_load_path_candidate}. Error: {e}[/]")
        if args.base_model_hub_name:
            console.print(f"Attempting to load tokenizer from Hub: [cyan]{args.base_model_hub_name}[/]")
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.base_model_hub_name)
                tokenizer_source_info = f"Hub ([cyan]{args.base_model_hub_name}[/])"
                console.print(f"[green]✓ Tokenizer loaded successfully from Hub.[/]")
            except Exception as hub_e:
                console.print(f"[bold red]❌ Failed to load tokenizer '{args.base_model_hub_name}' from Hub: {hub_e}[/]")
                console.print("[bold red]Aborting. Ensure the model path contains a tokenizer or provide a correct --base_model_hub_name.[/]")
                sys.exit(1)
        else:
            console.print("[bold red]❌ Cannot load tokenizer. Not found locally and no --base_model_hub_name provided. Aborting.[/]")
            sys.exit(1)

    if tokenizer is None: # Should not happen if logic above is correct, but as a safeguard
        console.print("[bold red]❌ Tokenizer failed to initialize. Aborting.[/]")
        sys.exit(1)

    # --- Determine Max Length ---
    effective_max_length = args.max_length # Use user override if provided
    if effective_max_length is None:
        # Try to infer from tokenizer config first
        effective_max_length = getattr(tokenizer, 'model_max_length', None)
        if effective_max_length is None or effective_max_length > 10000: # Check for unreasonably large values sometimes present
            # Fallback: Try to infer from model config of the first model
            try:
                config = AutoConfig.from_pretrained(model_paths_to_load[0])
                effective_max_length = getattr(config, 'max_position_embeddings', 512)
                if effective_max_length is None or effective_max_length > 10000: effective_max_length = 512
            except Exception:
                effective_max_length = 512 # Ultimate fallback
            console.print(f"[dim]Inferred max_length from config/default: {effective_max_length}[/]")
        else:
             console.print(f"[dim]Inferred max_length from tokenizer: {effective_max_length}[/]")
    else:
        console.print(f"[dim]User specified max_length: {effective_max_length}[/]")

    console.print(f"Using effective max sequence length: [yellow]{effective_max_length}[/]")

    # --- Create Dataset & Loader ---
    console.print("Instantiating Dataset and DataLoader...")
    dataset = TextDataset(df['text'].tolist(), df['id'].tolist(), tokenizer, max_length=effective_max_length)
    # Use pin_memory if CUDA is available
    pin_mem = torch.cuda.is_available()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    ids_order = df['id'].tolist() # Preserve original ID order for output mapping
    console.print(f"[green]✓ DataLoader ready (Batch size: {args.batch_size}, Pin memory: {pin_mem}).[/]")

    # --- Configuration Summary ---
    mode = "Ensemble" if is_ensemble_mode else "Single Model"
    model_source_display = cv_base_dir_path if is_ensemble_mode else single_model_path
    num_models_str = f" ({len(model_paths_to_load)} models)" if is_ensemble_mode else ""

    summary_table = Table(title="🚀 Configuration Summary", box=box.ROUNDED, show_header=False, title_style="bold green")
    summary_table.add_row("Mode", f"[bold cyan]{mode}[/]")
    summary_table.add_row("Model Source(s)", f"[yellow]{model_source_display}{num_models_str}[/]")
    summary_table.add_row("Tokenizer Source", tokenizer_source_info)
    summary_table.add_row("Input Data", f"[magenta]{input_csv_path}[/]")
    summary_table.add_row("Output CSV", f"[blue]{output_csv_path}[/]")
    summary_table.add_row("Output ZIP", f"[blue]{output_zip_path}[/]")
    summary_table.add_row("Max Length", f"[yellow]{effective_max_length}[/]")
    summary_table.add_row("Batch Size", f"[yellow]{args.batch_size}[/]")
    summary_table.add_row("Device", f"[bold blue]{device}[/]")
    summary_table.add_row("Eval Labels Present", "[green]Yes[/]" if gold_labels_present else "[yellow]No[/]")
    summary_table.add_row("Error Samples to Show", str(args.show_errors_count) if gold_labels_present else "[dim]N/A[/]")
    console.print(summary_table)

    # --- Inference Loop ---
    all_fold_probs_np = [] # List to store probability arrays from each model
    console.print(f"\n[bold]🚀 Running Inference...[/]")

    for i, model_p in enumerate(model_paths_to_load):
        model_id_str = f"(Model {i+1}/{len(model_paths_to_load)})" if is_ensemble_mode else ""
        console.print(f"--- Loading {model_id_str} from [yellow]{os.path.basename(model_p)}[/] ---")
        model = None # Ensure model is None before loading attempt
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_p).to(device).eval()
            console.print(f"[green]✓ Model {model_id_str} loaded to {device}.[/]")
        except Exception as e:
            console.print(f"[bold red]❌ Error loading model {model_p}: {e}. Skipping this model.[/]")
            continue # Skip to the next model if loading fails

        # Pre-allocate numpy array for this fold's probabilities (CPU)
        # Using NaN as default helps identify unprocessed batches/items if errors occur
        fold_probs_current = np.full(len(dataset), np.nan, dtype=np.float32)
        # Map batch index back to original dataset index
        processed_indices_map = [] # Store original indices processed in each batch

        with torch.no_grad():
            start_index = 0
            for batch in track(loader, description=f"Predicting {model_id_str}...", console=console, transient=False):
                batch_size_current = len(batch["id"])
                end_index = start_index + batch_size_current
                # Get original indices for this batch
                current_original_indices = list(range(start_index, end_index))

                input_ids = batch["input_ids"].to(device, non_blocking=pin_mem)
                attention_mask = batch["attention_mask"].to(device, non_blocking=pin_mem)

                # Basic check: If all input_ids in a row are padding token (usually 0 or 1), it might be a dummy from tokenization error
                is_potentially_dummy = torch.all(input_ids == tokenizer.pad_token_id, dim=1) if hasattr(tokenizer, 'pad_token_id') else torch.all(input_ids == 0, dim=1)

                if torch.all(is_potentially_dummy):
                     # console.print(f"\n[yellow]Skipping batch starting at index {start_index} - all items appear to be dummy/padding.[/]")
                     start_index = end_index
                     continue # Skip this batch entirely

                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Calculate probabilities based on logits shape
                    if logits.shape[1] >= 2: # Standard binary/multi-class
                        probs = torch.softmax(logits, dim=1)[:, 1] # Probability of class 1
                    elif logits.shape[1] == 1: # Single logit output (often regression treated as binary)
                        probs = torch.sigmoid(logits).squeeze(-1)
                    else: # Unexpected shape
                        console.print(f"\n[yellow]Unexpected logits shape {logits.shape} for batch starting {start_index}. Assigning 0 probability.[/]")
                        probs = torch.zeros(batch_size_current, device=device) # Assign zero probability

                    # Move probabilities to CPU and convert to numpy
                    batch_probs_np = probs.cpu().numpy().astype(np.float32)

                    # Place probabilities into the correct positions in fold_probs_current
                    # Handle cases where some items might have been dummy padding
                    valid_mask = ~is_potentially_dummy.cpu().numpy()
                    valid_indices_in_batch = np.array(current_original_indices)[valid_mask]
                    valid_probs = batch_probs_np[valid_mask]

                    if len(valid_indices_in_batch) > 0:
                         fold_probs_current[valid_indices_in_batch] = valid_probs
                         # Any dummy items will remain NaN

                except Exception as pred_e:
                     console.print(f"\n[bold red]❌ Error during prediction for batch starting at index {start_index}: {pred_e}[/]")
                     # Probabilities for this batch remain NaN

                start_index = end_index # Move to next starting index

        # After processing all batches for this model, check for NaNs (unprocessed items)
        num_nans = np.isnan(fold_probs_current).sum()
        if num_nans > 0:
             console.print(f"[yellow]⚠️ Model {model_id_str}: {num_nans} items have NaN probability (likely due to tokenization or batch errors). Filling with 0.5.[/]")
             fold_probs_current = np.nan_to_num(fold_probs_current, nan=0.5) # Replace NaN with 0.5

        all_fold_probs_np.append(fold_probs_current)
        console.print(f"[green]✓ Predictions collected for model {model_id_str}.[/]")

        # Clean up GPU memory
        del model, outputs, logits, probs, batch_probs_np, fold_probs_current
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Post-Inference Checks ---
    if not all_fold_probs_np:
        console.print("[bold red]❌ No predictions were generated from any model. Aborting.[/]")
        sys.exit(1)

    num_models_contributed = len(all_fold_probs_np)
    if is_ensemble_mode and num_models_contributed < len(model_paths_to_load):
         console.print(f"[yellow]⚠️ Only {num_models_contributed} out of {len(model_paths_to_load)} models successfully generated predictions for the ensemble.[/]")
    if num_models_contributed == 0: # Should be caught by the check above, but for safety
         console.print("[bold red]❌ No models contributed predictions. Aborting.[/]")
         sys.exit(1)

    # --- Aggregation and Final Thresholding ---
    final_probs = None
    final_threshold = 0.5 # Default final threshold

    console.print("\n[bold cyan]Processing Final Probabilities and Threshold...[/]")
    if is_ensemble_mode:
        console.print(f"Combining probabilities from {num_models_contributed} models (averaging)...")
        try:
            # Stack the numpy arrays and calculate the mean along axis 0
            all_probs_stack = np.stack(all_fold_probs_np, axis=0)
            final_probs = np.mean(all_probs_stack, axis=0)
            console.print("[green]✓ Ensemble probabilities averaged.[/]")
        except ValueError as avg_e:
             console.print(f"[bold red]❌ Error averaging probabilities (likely shape mismatch): {avg_e}. Aborting.[/]")
             sys.exit(1)

        # Determine final threshold for the ensemble
        if args.final_threshold is not None:
            final_threshold = args.final_threshold
            console.print(f"Using user-specified final threshold: [yellow]{final_threshold:.4f}[/]")
        elif gold_labels_present and gold_map is not None:
            console.print("Optimizing threshold based on averaged ensemble probabilities and available gold labels...")
            try:
                # Get true labels in the same order as predictions/ids_order
                # Use -1 for IDs not found in gold_map (e.g., test set or missing labels)
                y_true_ordered_list = [gold_map.get(i, -1) for i in ids_order]
                y_true_ordered = np.array(y_true_ordered_list)

                # Filter probabilities and true labels where true label is valid (not -1)
                valid_indices_final = y_true_ordered != -1
                y_true_filtered = y_true_ordered[valid_indices_final]
                final_probs_filtered = final_probs[valid_indices_final]

                if len(y_true_filtered) > 0:
                    optimal_threshold, best_ensemble_f1 = find_optimal_threshold(y_true_filtered, final_probs_filtered)
                    # Add simple check for extreme threshold derived from optimization
                    if best_ensemble_f1 > 0.1 and (optimal_threshold <= 0.05 or optimal_threshold >= 0.95):
                         console.print(f"[yellow]⚠️ Optimized ensemble threshold is extreme ({optimal_threshold:.4f}, F1={best_ensemble_f1:.4f}). Consider results carefully or use --final_threshold.[/]")
                    final_threshold = optimal_threshold
                    console.print(f"Optimal threshold found for ensemble: [yellow]{final_threshold:.4f}[/] (Achieved F1={best_ensemble_f1:.4f} on available labels)")
                else:
                    final_threshold = 0.5 # Fallback if no valid labels after filtering
                    console.print("[yellow]No valid gold labels available for optimization after filtering. Using default threshold 0.5[/]")
            except Exception as opt_e:
                final_threshold = 0.5 # Fallback on error
                console.print(f"[red]Error during ensemble threshold optimization: {opt_e}. Using default 0.5[/]")
        else:
            # No labels, no specific threshold: Use average of individuals thresholds loaded earlier
            if individual_thresholds:
                # Filter out any potential None values if a threshold failed to load (should be default 0.5 though)
                valid_thresholds = [t for t in individual_thresholds if t is not None]
                if valid_thresholds:
                    final_threshold = np.mean(valid_thresholds)
                    console.print(f"Using average of individual model thresholds: [yellow]{final_threshold:.4f}[/] (No gold labels for optimization or --final_threshold specified)")
                else:
                    final_threshold = 0.5
                    console.print("[yellow]Could not calculate average of individual thresholds. Using default 0.5[/]")
            else: # Should not happen if loop ran
                 final_threshold = 0.5
                 console.print("[yellow]Using default threshold 0.5 (No gold labels or individual thresholds available)[/]")

    else: # Single Model Mode
        final_probs = all_fold_probs_np[0] # There's only one array of probabilities
        if args.final_threshold is not None:
             final_threshold = args.final_threshold
             console.print(f"Using user-specified final threshold: [yellow]{final_threshold:.4f}[/]")
        else:
             # Use the threshold loaded for this specific model (stored in individual_thresholds[0])
             final_threshold = individual_thresholds[0]
             console.print(f"Using threshold loaded from model/default: [yellow]{final_threshold:.4f}[/]")

    # Apply final threshold to get predictions
    console.print(f"Applying final threshold {final_threshold:.4f} to probabilities...")
    final_predictions = (final_probs >= final_threshold).astype(int)

    # --- Save Predictions ---
    console.print("\n💾 [bold]Saving Final Predictions...[/]")
    pred_df = None # Initialize pred_df
    try:
        if len(ids_order) != len(final_predictions):
             # This indicates a major issue, likely in dataset/dataloader or result aggregation
             console.print(f"[bold red]❌ Critical Error: Length mismatch between original IDs ({len(ids_order)}) and final predictions ({len(final_predictions)}). Cannot save.[/]")
             sys.exit(1)
        # Create DataFrame with original IDs and final predictions
        pred_df = pd.DataFrame({"id": ids_order, "label": final_predictions})
        pred_df.to_csv(output_csv_path, index=False, header=True)
        console.print(f"[green]✅ Predictions saved to:[/] {output_csv_path}")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to save predictions to {output_csv_path}: {e}[/]")
        sys.exit(1) # Exit if saving fails

    # --- Create Submission Zip ---
    console.print("\n📦 [bold]Creating Submission ZIP...[/]")
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Use arcname to ensure the CSV is at the root of the zip with the correct name
            zipf.write(output_csv_path, arcname=output_csv_name)
        console.print(f"[green]✅ Submission ZIP created:[/] {output_zip_path}")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to create ZIP file {output_zip_path}: {e}[/]")
        # Continue script even if zipping fails

    # --- Classification Report & Error Analysis (only if gold labels were processed) ---
    if gold_labels_present and gold_map is not None and pred_df is not None:
        console.print("\n📊 [bold magenta]Final Evaluation Report[/bold magenta] (on available labels)\n")
        try:
            # Reconstruct y_true corresponding to predictions safely using the gold_map
            # Assign -1 if an ID from predictions is not in the gold_map (e.g., test set items)
            y_true_final_list = [gold_map.get(i, -1) for i in ids_order]
            y_true_final_full = np.array(y_true_final_list)

            # Filter predictions and true labels where true label is valid (not -1)
            valid_eval_indices = y_true_final_full != -1
            y_true_eval = y_true_final_full[valid_eval_indices]
            # Use the corresponding final_predictions
            y_pred_eval = final_predictions[valid_eval_indices]

            if len(y_true_eval) == 0:
                 console.print("[yellow]⚠️ No samples with valid gold labels found for evaluation after filtering.[/]")
            else:
                console.print(f"Evaluating on {len(y_true_eval)} samples with valid gold labels.")
                report = classification_report(y_true_eval, y_pred_eval, output_dict=True, digits=4, zero_division=0)

                # Display Report Table
                report_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
                report_table.add_column("Label", style="dim", justify="center")
                report_table.add_column("Precision", justify="right")
                report_table.add_column("Recall", justify="right")
                report_table.add_column("F1-score", justify="right")
                report_table.add_column("Support", style="dim", justify="right")

                # Determine labels present in either true or predicted eval sets
                present_labels = sorted(list(np.unique(np.concatenate((y_true_eval, y_pred_eval)))))
                labels_in_report_keys = [str(l) for l in present_labels] # Keys in report dict are strings

                for label_str in labels_in_report_keys:
                    if label_str in report: # Check if the label exists in the report dictionary
                        metrics = report[label_str]
                        report_table.add_row(
                            label_str,
                            f"{metrics['precision']:.4f}",
                            f"{metrics['recall']:.4f}",
                            f"{metrics['f1-score']:.4f}",
                            f"{int(metrics['support'])}"
                        )
                    # else: The label might be present in y_true/y_pred but not have metrics if support is 0 for TP/FP/FN combinations

                # Add averages
                report_table.add_section()
                for avg_type in ["macro avg", "weighted avg"]:
                    if avg_type in report:
                        metrics = report[avg_type]
                        name = avg_type.replace(" avg", " Avg")
                        report_table.add_row(f"[bold]{name}[/]", f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1-score']:.4f}", f"{int(metrics['support'])}")

                # Add accuracy
                if "accuracy" in report:
                    accuracy = report["accuracy"]
                    total_support = int(report.get("weighted avg", {}).get("support", len(y_true_eval))) # Get total support
                    report_table.add_section()
                    report_table.add_row("[bold]Accuracy[/]", "", "", f"[bold]{accuracy:.4f}[/]", f"{total_support}")
                console.print(report_table)

                # Display Confusion Matrix
                console.print("\n🎯 [bold blue]Confusion Matrix[/bold blue]\n")
                # Use the labels actually present in the evaluation set for CM
                cm_labels = sorted(list(np.unique(y_true_eval))) # Base labels on true values present
                if not cm_labels: cm_labels=[0, 1] # Fallback if only one class truly present

                cm = confusion_matrix(y_true_eval, y_pred_eval, labels=cm_labels)

                cm_table = Table(title="True \\ Predicted", box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
                cm_table.add_column("Actual", justify="center", style="bold dim")
                for pred_label in cm_labels: # Columns based on the same labels
                     cm_table.add_column(f"Pred {pred_label}", justify="center", style="cyan" if pred_label == 0 else "magenta")

                for i, true_label in enumerate(cm_labels):
                    # Ensure row corresponds correctly if cm shape differs slightly (e.g., only one class predicted)
                    row_values = cm[i] if i < cm.shape[0] else [0] * len(cm_labels)
                    cm_table.add_row(f"True {true_label}", *[str(count) for count in row_values])
                console.print(cm_table)

                # --- Error Analysis (Added from old script) ---
                console.print("\n🧐 [bold yellow]Error Analysis (Sample)[/bold yellow]\n")
                try:
                    # Merge original df (with text, labels) and prediction df
                    # Ensure 'id' column exists and has compatible types if necessary
                    analysis_df = pd.merge(
                        df[['id', 'text', 'labels']], # Select needed cols from original df
                        pred_df[['id', 'label']],      # Select needed cols from prediction df
                        on='id',
                        how='inner' # Only include IDs present in both (should be all if no errors)
                    )

                    # Filter out rows where the original label was not valid (e.g., NaN, non-numeric)
                    # Use the y_true_final_full array which already reflects valid/invalid gold labels (-1)
                    # Create a boolean mask based on IDs present in analysis_df
                    analysis_ids = analysis_df['id'].tolist()
                    original_indices = [ids_order.index(aid) for aid in analysis_ids]
                    valid_label_mask_for_analysis = y_true_final_full[original_indices] != -1
                    analysis_df = analysis_df[valid_label_mask_for_analysis].copy() # Filter analysis_df

                    # Ensure 'labels' (true) and 'label' (pred) are integer type for comparison
                    analysis_df['labels'] = analysis_df['labels'].astype(int)
                    analysis_df['label'] = analysis_df['label'].astype(int)

                    # Identify False Positives (True=0, Pred=1)
                    false_positives = analysis_df[(analysis_df["labels"] == 0) & (analysis_df["label"] == 1)]

                    # Identify False Negatives (True=1, Pred=0)
                    false_negatives = analysis_df[(analysis_df["labels"] == 1) & (analysis_df["label"] == 0)]

                    # Use the helper function to display errors
                    show_errors(false_positives, "[red]False Positives (Pred 1, True 0)", console, max_items=args.show_errors_count)
                    show_errors(false_negatives, "[magenta]False Negatives (Pred 0, True 1)", console, max_items=args.show_errors_count)

                except Exception as analysis_e:
                    console.print(f"[red]Could not perform error analysis step: {analysis_e}[/]")
                    import traceback
                    traceback.print_exc() # Print traceback for debugging merge/filter issues


        except Exception as report_e:
            console.print(f"[bold red]❌ Error generating final evaluation report/analysis: {report_e}[/]")
            import traceback
            traceback.print_exc()

    elif pred_df is not None: # Gold labels weren't present or processed
        console.print("\n[dim]Evaluation report and error analysis skipped as gold labels were not available or processed.[/]")

    console.print("\n[bold green]🏁 Script finished successfully![/]")

if __name__ == "__main__":
    main()