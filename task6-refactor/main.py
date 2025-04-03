# main.py
import os
import argparse
import logging
from src.data import DataProcessor
from src.dataset import VaccineDataset
from src.model import VAEMClassificationModel
from src.trainer import CustomVAEMTrainer
from src.utils import (
    set_global_seed,
    load_config,
    setup_logging,
    plot_learning_curves,
    plot_confusion_matrix
)
from transformers import TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import confusion_matrix

# Rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

def main(config_path: str = 'config.yaml'):
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    setup_logging(log_file=config['logging']['log_file'])
    logger = logging.getLogger(__name__)

    # Set global seed
    set_global_seed(config['experiment']['seed'])

    # Display header
    console.rule("[bold blue]💉 SMM4H 2025 — VAEM Detection Fine-tuning[/bold blue]")
    console.print(Panel.fit(
        f"[cyan]Experiment:[/] [bold]{config['experiment']['name']}[/]\n"
        f"[green]Seed:[/] {config['experiment']['seed']}   "
        f"[magenta]K-Folds:[/] {config['training']['k_folds']}   "
        f"[yellow]Model:[/] {config['model']['name']}",
        title="Configuration", border_style="green"
    ))

    # Load data
    train_df, test_df = DataProcessor.load_data(
        config['data']['train_path'],
        config['data']['test_path']
    )

    if config['data']['augmentation']:
        console.print("[bold yellow]⚙️ Performing data augmentation...[/bold yellow]")
        train_df = DataProcessor.data_augmentation(train_df)

    # Split data
    train_splits = DataProcessor.stratified_kfold_split(
        train_df,
        n_splits=config['training']['k_folds']
    )

    fold_results = []

    for fold, (train_fold, valid_fold) in enumerate(train_splits, 1):
        console.rule(f"[bold yellow]🌀 Fold {fold} - Training[/bold yellow]")

        model = VAEMClassificationModel.create_model(
            model_name=config['model']['name'],
            dropout_rate=config['model']['dropout']
        )
        tokenizer = VAEMClassificationModel.load_tokenizer(
            model_name=config['model']['name']
        )

        train_dataset = VaccineDataset(train_fold, tokenizer)
        valid_dataset = VaccineDataset(valid_fold, tokenizer)

        training_args = TrainingArguments(
            output_dir=f"./results/fold_{fold}",
            num_train_epochs=int(config['training']['epochs']),
            per_device_train_batch_size=int(config['training']['batch_size']),
            per_device_eval_batch_size=int(config['training']['batch_size']),
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_beta",
            fp16=True
        )

        trainer = CustomVAEMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=CustomVAEMTrainer.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

        best_threshold, best_fbeta = trainer.adaptive_threshold_search(valid_dataset)

        eval_results = trainer.evaluate()
        fold_results.append({
            'fold': fold,
            'best_threshold': best_threshold,
            'best_fbeta': best_fbeta,
            **eval_results
        })

        try:
            plot_learning_curves(
                trainer.state.log_history,
                trainer.state.log_history,
                output_dir=f"./results/fold_{fold}/plots"
            )
            console.print(f"[green]📈 Learning curves saved to:[/] ./results/fold_{fold}/plots")
        except Exception as e:
            console.print(f"[red]⚠️ Could not plot learning curves for Fold {fold}[/] — {e}")

        console.print(
            f"[bold green]✅ Fold {fold} completed[/] — "
            f"F1β: [cyan]{best_fbeta:.4f}[/] | Threshold: [magenta]{best_threshold:.2f}[/]"
        )

    # Final summary
    console.rule("[bold green]📊 Cross-Validation Summary[/bold green]")
    summary_table = Table(title="Fold Results", box=box.ROUNDED)
    summary_table.add_column("Fold", justify="center")
    summary_table.add_column("Best F1β", justify="right")
    summary_table.add_column("Best Threshold", justify="right")
    summary_table.add_column("Eval Loss", justify="right")

    for result in fold_results:
        summary_table.add_row(
            str(result["fold"]),
            f"{result['best_fbeta']:.4f}",
            f"{result['best_threshold']:.2f}",
            f"{result.get('eval_loss', '—'):.4f}"
        )

    console.print(summary_table)
    console.rule("[bold blue]🏁 Fine-tuning complete![/bold blue]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()
    main(args.config)
