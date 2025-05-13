import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import zipfile
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
from collections import Counter # For majority voting
import logging # For better logging
from tqdm.auto import tqdm # Import tqdm for progress bars
import gc # Import garbage collector

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- File Configuration ---
DEV_FILE = "data/test_data_SMM4H_2025_Task_1_no_labels.csv"

# --- Ensemble Configuration ---
MODEL_PATHS = [
    # --- ADD YOUR RELATIVE PATHS HERE ---
    "easy_xlmr_large_256/checkpoint-5844"#, Finally, we use only one model for inference here
]
# --- Batch Size to use for inference (can be larger now) ---
INFERENCE_BATCH_SIZE = 32

# Optional: Weights for weighted voting
# MODEL_WEIGHTS = [1.0] * len(MODEL_PATHS)
# ENSEMBLE_THRESHOLD = 0.5

# --- Output Configuration ---
OUTPUT_DIR = "ensemble_submission_output_load_unload" # Different folder name to avoid confusion
SUBMISSION_CSV_NAME = "predictions_task1.csv"
SUBMISSION_ZIP_NAME = "submission_task1.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Determining script directory ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Detected script directory: {script_dir}")
except NameError:
    script_dir = os.getcwd()
    logging.warning(f"Could not determine script directory via __file__, using current working directory: {script_dir}.")
    logging.warning("Ensure this current directory is indeed the 'task1' folder.")

# --- Device Check ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
if not MODEL_PATHS:
    raise ValueError("MODEL_PATHS list is empty.")

# --- Loading data file ---
data_file_path = os.path.join(script_dir, DEV_FILE)
logging.info(f"--- Loading data file from: {data_file_path} ---")
if not os.path.exists(data_file_path):
     raise FileNotFoundError(f"Error: Data file not found at {data_file_path}")
try:
    df = pd.read_csv(data_file_path)
    if "text" not in df.columns:
        raise ValueError("The 'text' column is missing.")
    initial_rows = len(df)
    df.dropna(subset=["text"], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logging.warning(f"{dropped_rows} rows removed because 'text' was empty/NA.")
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Number of examples loaded: {len(df)}")
except Exception as e:
    logging.error(f"Error during data loading/preprocessing: {e}", exc_info=True)
    raise
texts_to_predict = df["text"].astype(str).tolist()


# --- Batch prediction function (remains identical) ---
def predict_batch(texts, model, tokenizer, threshold, batch_size=16, device="cpu"):
    """Performs batch prediction with a tqdm progress bar."""
    model.eval()
    all_preds = []
    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="   Batch Prediction", leave=False, unit="batch"):
        batch_texts = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            batch_preds = (probs >= threshold).astype(int)
            all_preds.extend(batch_preds)
        except Exception as e:
            logging.error(f"     Error during prediction (batch index {i}): {e}", exc_info=False)
            all_preds.extend([0] * len(batch_texts))
            all_probs.extend([0.0] * len(batch_texts))
            logging.warning(f"     -> Default predictions (0) used for this batch.")
    return all_preds, all_probs


# --- Main loop: Load, Predict, Unload ---
all_model_preds = []
all_model_probs = []
loaded_model_paths_history = [] # Just to keep track of processed models

logging.info(f"--- Starting predictions (Load/Unload) with Batch Size = {INFERENCE_BATCH_SIZE} ---")

# Use tqdm to track progress over different models
for i, relative_model_path in enumerate(tqdm(MODEL_PATHS, desc="Model Load/Inference/Unload", unit="model")):

    absolute_model_path = os.path.join(script_dir, relative_model_path)
    logging.info(f"--- Processing Model {i+1}/{len(MODEL_PATHS)}: '{relative_model_path}' ---")

    if not os.path.exists(absolute_model_path):
        logging.warning(f"  -> Path not found, model skipped: {absolute_model_path}")
        # Add empty or default predictions to maintain structure?
        # For now, we just ignore, but be careful with voting if some models are missing.
        continue

    model = None # Ensure variables are null before loading
    tokenizer = None
    threshold = 0.5 # Default threshold

    try:
        # 1. LOAD model and tokenizer
        logging.info(f"  1. Loading from: {absolute_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(absolute_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(absolute_model_path)
        model.to(device)
        model.eval()

        # Load optimized threshold
        parent_dir = os.path.dirname(absolute_model_path)
        threshold_path = os.path.join(parent_dir, "threshold.txt")
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path) as f:
                    threshold = float(f.read().strip())
                logging.info(f"     Threshold loaded: {threshold}")
            except Exception as e:
                logging.warning(f"     Could not read threshold ({threshold_path}), using default (0.5). Error: {e}")
        else:
            logging.info(f"     Threshold file not found ({threshold_path}), using default (0.5).")

        # 2. PREDICTION
        logging.info(f"  2. Prediction (Batch size: {INFERENCE_BATCH_SIZE}, Threshold: {threshold:.4f})")
        model_preds, model_probs = predict_batch(
            texts_to_predict,
            model,
            tokenizer,
            threshold,
            batch_size=INFERENCE_BATCH_SIZE, # Using configured batch size
            device=device
        )
        all_model_preds.append(model_preds)
        all_model_probs.append(model_probs)
        loaded_model_paths_history.append(relative_model_path) # Model processed successfully

    except Exception as e:
        logging.error(f"  -> ERROR while processing model '{relative_model_path}': {e}", exc_info=True)
        # Do not add predictions if loading or inference fails

    finally:
        # 3. UNLOAD (important to do this even in case of partial error)
        logging.info(f"  3. Unloading model '{relative_model_path}'")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        # Call garbage collector and empty CUDA cache
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            # Optional: check memory after cleaning
            # logging.debug(f"     GPU memory allocated after cleaning: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            # logging.debug(f"     GPU memory reserved after cleaning: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")
        logging.info(f"  -> Model '{relative_model_path}' unloaded.")


# Check if any predictions were generated
if not all_model_preds:
    raise RuntimeError("FATAL: No predictions were generated. Check error logs for each model.")
logging.info(f"--- Individual predictions completed for {len(all_model_preds)} models ---")


# --- Ensemble Voting ---
logging.info("--- Combining predictions by Majority Vote ---")
final_predictions = []

# Ensure all prediction lists have the correct length
if not all(len(preds) == len(texts_to_predict) for preds in all_model_preds):
      logging.error("Inconsistency in the number of predictions generated by successfully processed models.")
      raise RuntimeError("The number of predictions per model does not match the number of texts. Check previous errors.")

num_predictions = len(texts_to_predict)

for i in tqdm(range(num_predictions), desc="Majority Voting", unit="sample"):
    try:
        # Get predictions from models that succeeded
        sample_preds = [preds[i] for preds in all_model_preds]
    except IndexError:
         logging.error(f"Index error for sample {i}.")
         final_predictions.append(0)
         continue

    # --- Simple Majority Vote ---
    if not sample_preds: # Case where no model succeeded for this index (should not happen with the previous check)
        logging.warning(f"No predictions available for sample {i}. Using 0 by default.")
        final_predictions.append(0)
        continue

    counts = Counter(sample_preds)
    if len(counts) > 1 and counts.most_common(2)[0][1] == counts.most_common(2)[1][1]:
         mean_vote = np.mean(sample_preds)
         majority_vote = 1 if mean_vote >= 0.5 else 0
    elif len(counts) == 1:
         majority_vote = counts.most_common(1)[0][0]
    else:
         majority_vote = counts.most_common(1)[0][0]
    final_predictions.append(majority_vote)



# Add final predictions to DataFrame
if len(final_predictions) == len(df):
    df["predicted_label"] = final_predictions
    logging.info("Majority vote completed and predictions added to DataFrame.")
else:
    logging.error(f"Number of final predictions ({len(final_predictions)}) != DataFrame size ({len(df)}).")
    raise RuntimeError("Critical error when assembling final predictions.")


# --- Evaluation (if labels present) ---
if "label" in df.columns:
    logging.info("--- Evaluating ensemble results ---")
    try:
        df['label'] = df['label'].astype(int)
        df['predicted_label'] = df['predicted_label'].astype(int)
        evaluation_possible = True
    except Exception as e:
         logging.error(f"Could not convert 'label'/'predicted_label' columns to integers: {e}")
         evaluation_possible = False
    if evaluation_possible:
        y_true = df['label']
        y_pred = df['predicted_label']
        precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        logging.info("--- Overall Evaluation Metrics (Positive Class = 1) ---")
        logging.info(f"  F1-score (Pos):  {f1_pos:.4f}  <-- Main Metric")
        logging.info(f"  Precision (Pos): {precision_pos:.4f}")
        logging.info(f"  Recall (Pos):    {recall_pos:.4f}")
        logging.info(f"  Accuracy:        {accuracy:.4f}")
        cm_overall = confusion_matrix(y_true, y_pred, labels=[0, 1])
        logging.info("  Overall Confusion Matrix:")
        logging.info(f"  {cm_overall}")
        if cm_overall.size == 4:
             tn, fp, fn, tp = cm_overall.ravel()
             logging.info(f"    [[TN={tn}  FP={fp}]")
             logging.info(f"     [FN={fn}  TP={tp}]]")
        if "language" in df.columns:
            pass
        else:
             logging.warning("Column 'language' not found, detailed evaluation by language skipped.")
elif "label" not in df.columns:
     logging.info("Column 'label' not found. Evaluation skipped.")


# --- Saving final predictions ---
logging.info("--- Saving final predictions ---")
if "id" not in df.columns:
      raise ValueError("Column 'id' missing.")
if "predicted_label" not in df.columns:
      raise ValueError("Column 'predicted_label' missing.")
submission_df = df[["id", "predicted_label"]]
csv_path = os.path.join(OUTPUT_DIR, SUBMISSION_CSV_NAME)
zip_path = os.path.join(OUTPUT_DIR, SUBMISSION_ZIP_NAME)
try:
    submission_df.to_csv(csv_path, index=False)
    logging.info(f"Predictions saved to: {csv_path}")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=SUBMISSION_CSV_NAME)
    logging.info(f"CSV file '{SUBMISSION_CSV_NAME}' compressed into: {zip_path}")
except Exception as e:
    logging.error(f"Error during saving: {e}", exc_info=True)

logging.info("========== Script finished ==========")
logging.info(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
logging.info(f"Submission file generated: {zip_path}")