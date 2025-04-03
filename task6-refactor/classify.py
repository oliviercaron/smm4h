import os
import sys
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import zipfile
from sklearn.metrics import classification_report, confusion_matrix
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Setup Rich console
console = Console()

# ---------------------
# Dataset class
# ---------------------
class TextDataset(Dataset):
    def __init__(self, texts, ids, tokenizer, max_length=512):
        self.texts = texts
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        id_ = self.ids[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "id": id_
        }

# ---------------------
# Main function
# ---------------------
def main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/] python classify.py <model_name> [data_file.csv]")
        sys.exit(1)

    model_name = sys.argv[1]
    input_filename = sys.argv[2] if len(sys.argv) >= 3 else "test_data.csv"
    input_csv = os.path.join("data", input_filename)

    model_path = os.path.join("models", model_name)
    output_dir = "results"
    output_csv = os.path.join(output_dir, "prediction_task6.csv")
    output_zip = os.path.join(output_dir, "submission.zip")
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        console.print(f"[bold red]❌ Model folder not found:[/] {model_path}")
        sys.exit(1)

    if not os.path.exists(input_csv):
        console.print(f"[bold red]❌ Input file not found:[/] {input_csv}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    console.print(Panel.fit(f"🔍 Loading model from [bold cyan]{model_path}[/]", title="Model", border_style="cyan"))
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load data
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns or 'id' not in df.columns:
        console.print("[bold red]❌ CSV must contain 'id' and 'text' columns.[/]")
        sys.exit(1)

    dataset = TextDataset(df['text'].tolist(), df['id'].tolist(), tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Inference
    predictions = []
    ids = []

    console.print(f"[bold]🚀 Running inference on[/] [green]{input_csv}[/] using [blue]{device}[/]...\n")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)
            ids.extend([int(i) for i in batch["id"]])

    # Save predictions
    pred_df = pd.DataFrame({"id": ids, "label": predictions})
    pred_df.to_csv(output_csv, index=False)
    console.print(f"[green]✅ Predictions saved to:[/] {output_csv}")

    # Zip
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_csv, arcname="prediction_task6.csv")
    console.print(f"[green]📦 Submission ZIP created:[/] {output_zip}")

    # Classification report
    if "valid" in input_filename.lower():
        console.print("\n📊 [bold magenta]Classification Report (Validation Set)[/bold magenta]\n")

        gold_df = pd.read_csv(input_csv)
        if "labels" not in gold_df.columns:
            console.print("[yellow]⚠️ No 'labels' column found in validation file. Skipping report.[/]")
        else:
            y_true = gold_df.set_index("id").loc[pred_df["id"]]["labels"]
            y_pred = pred_df["label"]
            report = classification_report(y_true, y_pred, output_dict=True, digits=4)

            table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
            table.add_column("Label", justify="center")
            table.add_column("Precision", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("F1-score", justify="right")
            table.add_column("Support", justify="right")

            for label in ["0", "1", "macro avg", "weighted avg"]:
                m = report[label]
                name = label if "avg" not in label else label.replace(" avg", " Avg")
                table.add_row(name,
                            f"{m['precision']:.4f}",
                            f"{m['recall']:.4f}",
                            f"{m['f1-score']:.4f}",
                            f"{int(m['support']) if 'support' in m else '-'}")

            accuracy = report["accuracy"]
            table.add_row("—", "—", "—", "—", "—")
            table.add_row("Accuracy", "", "", f"{accuracy:.4f}", str(int(sum(report[l]["support"] for l in ["0", "1"]))))

            console.print(table)

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            # Render confusion matrix as a Rich table
            console.print("\n🎯 [bold blue]Confusion Matrix[/bold blue]\n")

            cm_table = Table(title="True vs Predicted", box=box.SIMPLE_HEAVY)
            cm_table.add_column("Actual \\ Pred", justify="center", style="bold")
            cm_table.add_column("0", justify="center", style="cyan")
            cm_table.add_column("1", justify="center", style="magenta")

            cm_table.add_row("0", str(cm[0][0]), str(cm[0][1]))
            cm_table.add_row("1", str(cm[1][0]), str(cm[1][1]))

            console.print(cm_table)

            # 🔎 Show misclassified examples (false positives and false negatives)
            console.print("\n🧐 [bold yellow]Error Analysis[/bold yellow]\n")

            merged_df = gold_df.set_index("id").loc[pred_df["id"]].copy()
            merged_df["predicted"] = pred_df["label"].values

            # Faux positifs : vrai label = 0, prédit = 1
            false_positives = merged_df[(merged_df["labels"] == 0) & (merged_df["predicted"] == 1)]

            # Faux négatifs : vrai label = 1, prédit = 0
            false_negatives = merged_df[(merged_df["labels"] == 1) & (merged_df["predicted"] == 0)]

            def show_errors(df, title, max_items=5):
                console.rule(title)
                for i, row in df.head(max_items).iterrows():
                    console.print(f"[bold cyan]ID:[/] {i}")
                    console.print(f"[red]TRUE:[/] {row['labels']}  |  [green]PRED:[/] {row['predicted']}")
                    console.print(f"[white]{row['text']}\n")

            if not false_positives.empty:
                show_errors(false_positives, "[red]False Positives (pred 1 but should be 0)")

            if not false_negatives.empty:
                show_errors(false_negatives, "[red]False Negatives (pred 0 but should be 1)")

if __name__ == "__main__":
    main()
