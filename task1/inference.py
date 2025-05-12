import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import zipfile
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
from collections import Counter # Pour le vote majoritaire
import logging # Pour un meilleur logging
from tqdm.auto import tqdm # Importation de tqdm pour les barres de progression
import gc # Importer le garbage collector

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration des Fichiers ---
DEV_FILE = "data/test_data_SMM4H_2025_Task_1_no_labels.csv"

# --- Configuration de l'Ensemble ---
MODEL_PATHS = [
    # --- AJOUTE TES CHEMINS RELATIFS ICI ---
    "easy_xlmr_large_256/checkpoint-5844"#, Finally, we use only one model for inference here
]
# --- Taille de Batch à utiliser pour l'inférence (peut être plus grande maintenant) ---
INFERENCE_BATCH_SIZE = 32

# Optionnel: Poids pour le vote pondéré
# MODEL_WEIGHTS = [1.0] * len(MODEL_PATHS)
# ENSEMBLE_THRESHOLD = 0.5

# --- Configuration des sorties ---
OUTPUT_DIR = "ensemble_submission_output_load_unload" # Nom de dossier différent pour éviter confusion
SUBMISSION_CSV_NAME = "predictions_task1.csv"
SUBMISSION_ZIP_NAME = "submission_task1.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Détermination du répertoire du script ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging.info(f"Répertoire du script détecté: {script_dir}")
except NameError:
    script_dir = os.getcwd()
    logging.warning(f"Impossible de déterminer le répertoire du script via __file__, utilisation du répertoire courant: {script_dir}.")
    logging.warning("Assurez-vous que ce répertoire courant est bien le dossier 'task1'.")

# --- Vérification du device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Utilisation du device: {device}")
if not MODEL_PATHS:
    raise ValueError("La liste MODEL_PATHS est vide.")

# --- Chargement du fichier de données ---
data_file_path = os.path.join(script_dir, DEV_FILE)
logging.info(f"--- Chargement du fichier de données depuis: {data_file_path} ---")
if not os.path.exists(data_file_path):
     raise FileNotFoundError(f"Erreur: Fichier de données non trouvé à {data_file_path}")
try:
    df = pd.read_csv(data_file_path)
    if "text" not in df.columns:
        raise ValueError("La colonne 'text' est manquante.")
    initial_rows = len(df)
    df.dropna(subset=["text"], inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        logging.warning(f"{dropped_rows} lignes supprimées car 'text' était vide/NA.")
    df.reset_index(drop=True, inplace=True)
    logging.info(f"Nombre d'exemples chargés: {len(df)}")
except Exception as e:
    logging.error(f"Erreur lors du chargement/prétraitement des données: {e}", exc_info=True)
    raise
texts_to_predict = df["text"].astype(str).tolist()


# --- Fonction de prédiction par batch (reste identique) ---
def predict_batch(texts, model, tokenizer, threshold, batch_size=16, device="cpu"):
    """Effectue la prédiction par batch avec une barre de progression tqdm."""
    model.eval()
    all_preds = []
    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Batch Prediction", leave=False, unit="batch"):
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
            logging.error(f"    Erreur durant la prédiction (batch index {i}): {e}", exc_info=False)
            all_preds.extend([0] * len(batch_texts))
            all_probs.extend([0.0] * len(batch_texts))
            logging.warning(f"    -> Prédictions par défaut (0) utilisées pour ce batch.")
    return all_preds, all_probs


# --- Boucle principale : Chargement, Prédiction, Déchargement ---
all_model_preds = []
all_model_probs = []
loaded_model_paths_history = [] # Juste pour garder une trace des modèles traités

logging.info(f"--- Début des prédictions (Chargement/Déchargement) avec Batch Size = {INFERENCE_BATCH_SIZE} ---")

# Utiliser tqdm pour suivre la progression sur les différents modèles
for i, relative_model_path in enumerate(tqdm(MODEL_PATHS, desc="Model Load/Inference/Unload", unit="model")):

    absolute_model_path = os.path.join(script_dir, relative_model_path)
    logging.info(f"--- Traitement Modèle {i+1}/{len(MODEL_PATHS)}: '{relative_model_path}' ---")

    if not os.path.exists(absolute_model_path):
        logging.warning(f"  -> Chemin non trouvé, modèle ignoré: {absolute_model_path}")
        # Ajouter des prédictions vides ou par défaut pour maintenir la structure ?
        # Pour l'instant on ignore juste, mais attention au vote si certains modèles manquent.
        continue

    model = None # S'assurer que les variables sont nulles avant de charger
    tokenizer = None
    threshold = 0.5 # Seuil par défaut

    try:
        # 1. CHARGEMENT du modèle et tokenizer
        logging.info(f"  1. Chargement depuis: {absolute_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(absolute_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(absolute_model_path)
        model.to(device)
        model.eval()

        # Charger le seuil optimisé
        parent_dir = os.path.dirname(absolute_model_path)
        threshold_path = os.path.join(parent_dir, "threshold.txt")
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path) as f:
                    threshold = float(f.read().strip())
                logging.info(f"     Seuil chargé: {threshold}")
            except Exception as e:
                logging.warning(f"     Impossible lire seuil ({threshold_path}), utilisation défaut (0.5). Erreur: {e}")
        else:
            logging.info(f"     Fichier seuil non trouvé ({threshold_path}), utilisation défaut (0.5).")

        # 2. PRÉDICTION
        logging.info(f"  2. Prédiction (Batch size: {INFERENCE_BATCH_SIZE}, Seuil: {threshold:.4f})")
        model_preds, model_probs = predict_batch(
            texts_to_predict,
            model,
            tokenizer,
            threshold,
            batch_size=INFERENCE_BATCH_SIZE, # Utilisation de la taille de batch configurée
            device=device
        )
        all_model_preds.append(model_preds)
        all_model_probs.append(model_probs)
        loaded_model_paths_history.append(relative_model_path) # Modèle traité avec succès

    except Exception as e:
        logging.error(f"  -> ERREUR lors du traitement du modèle '{relative_model_path}': {e}", exc_info=True)
        # Ne pas ajouter de prédictions si le chargement ou l'inférence échoue

    finally:
        # 3. DÉCHARGEMENT (important de le faire même en cas d'erreur partielle)
        logging.info(f"  3. Déchargement du modèle '{relative_model_path}'")
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        # Appeler le garbage collector et vider le cache CUDA
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            # Optionnel: vérifier la mémoire après nettoyage
            # logging.debug(f"     Mémoire GPU allouée après nettoyage: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
            # logging.debug(f"     Mémoire GPU réservée après nettoyage: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")
        logging.info(f"  -> Modèle '{relative_model_path}' déchargé.")


# Vérifier si des prédictions ont été générées
if not all_model_preds:
    raise RuntimeError("FATAL: Aucune prédiction n'a été générée. Vérifiez les logs d'erreur pour chaque modèle.")
logging.info(f"--- Prédictions individuelles terminées pour {len(all_model_preds)} modèles ---")


# --- Ensemble Voting ---
logging.info("--- Combinaison des prédictions par Vote Majoritaire ---")
final_predictions = []

# S'assurer que toutes les listes de prédictions ont la bonne longueur
if not all(len(preds) == len(texts_to_predict) for preds in all_model_preds):
     logging.error("Incohérence dans le nombre de prédictions générées par les modèles traités avec succès.")
     raise RuntimeError("Le nombre de prédictions par modèle ne correspond pas au nombre de textes. Vérifiez les erreurs précédentes.")

num_predictions = len(texts_to_predict)

for i in tqdm(range(num_predictions), desc="Majority Voting", unit="sample"):
    try:
        # Récupérer les prédictions des modèles qui ont réussi
        sample_preds = [preds[i] for preds in all_model_preds]
    except IndexError:
         logging.error(f"Erreur d'index pour l'échantillon {i}.")
         final_predictions.append(0)
         continue

    # --- Vote Majoritaire Simple ---
    if not sample_preds: # Cas où aucun modèle n'a réussi pour cet index (ne devrait pas arriver avec le check précédent)
        logging.warning(f"Aucune prédiction disponible pour l'échantillon {i}. Utilisation de 0 par défaut.")
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



# Ajouter les prédictions finales au DataFrame
if len(final_predictions) == len(df):
    df["predicted_label"] = final_predictions
    logging.info("Vote majoritaire terminé et prédictions ajoutées au DataFrame.")
else:
    logging.error(f"Nb prédictions finales ({len(final_predictions)}) != Taille DataFrame ({len(df)}).")
    raise RuntimeError("Erreur critique lors de l'assemblage des prédictions finales.")


# --- Évaluation (si labels présents) ---
if "label" in df.columns:
    logging.info("--- Évaluation des résultats de l'ensemble ---")
    try:
        df['label'] = df['label'].astype(int)
        df['predicted_label'] = df['predicted_label'].astype(int)
        evaluation_possible = True
    except Exception as e:
         logging.error(f"Impossible de convertir les colonnes 'label'/'predicted_label' en entiers: {e}")
         evaluation_possible = False
    if evaluation_possible:
        y_true = df['label']
        y_pred = df['predicted_label']
        precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        logging.info("--- Métriques Globales d'Évaluation (Classe Positive = 1) ---")
        logging.info(f"  F1-score (Pos):  {f1_pos:.4f}  <-- Métrique Principale")
        logging.info(f"  Précision (Pos): {precision_pos:.4f}")
        logging.info(f"  Rappel (Pos):    {recall_pos:.4f}")
        logging.info(f"  Accuracy:        {accuracy:.4f}")
        cm_overall = confusion_matrix(y_true, y_pred, labels=[0, 1])
        logging.info("  Matrice de Confusion Globale:")
        logging.info(f"  {cm_overall}")
        if cm_overall.size == 4:
             tn, fp, fn, tp = cm_overall.ravel()
             logging.info(f"    [[TN={tn}  FP={fp}]")
             logging.info(f"     [FN={fn}  TP={tp}]]")
        if "language" in df.columns:
            pass 
        else:
             logging.warning("Colonne 'language' non trouvée, évaluation détaillée par langue ignorée.")
elif "label" not in df.columns:
     logging.info("Colonne 'label' non trouvée. Évaluation ignorée.")


# --- Sauvegarde des prédictions finales ---
logging.info("--- Sauvegarde des prédictions finales ---")
if "id" not in df.columns:
      raise ValueError("Colonne 'id' manquante.")
if "predicted_label" not in df.columns:
      raise ValueError("Colonne 'predicted_label' manquante.")
submission_df = df[["id", "predicted_label"]]
csv_path = os.path.join(OUTPUT_DIR, SUBMISSION_CSV_NAME)
zip_path = os.path.join(OUTPUT_DIR, SUBMISSION_ZIP_NAME)
try:
    submission_df.to_csv(csv_path, index=False)
    logging.info(f"Prédictions sauvegardées dans: {csv_path}")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=SUBMISSION_CSV_NAME)
    logging.info(f"Fichier CSV '{SUBMISSION_CSV_NAME}' compressé dans: {zip_path}")
except Exception as e:
    logging.error(f"Erreur lors de la sauvegarde: {e}", exc_info=True)

logging.info("========== Script terminé ==========")
logging.info(f"Répertoire de sortie: {os.path.abspath(OUTPUT_DIR)}")
logging.info(f"Fichier de soumission généré: {zip_path}")