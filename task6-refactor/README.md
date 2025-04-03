# SMM4H 2025 Task 6: Vaccine Adverse Event Mention Detection

## Project Overview
This project addresses the challenge of detecting vaccine adverse event mentions (VAEM) in Reddit posts related to shingles vaccination.

## Features
- Advanced text classification using PubMedBERT
- Custom loss functions (Cross-Entropy, Focal Loss, Dice Loss)
- K-Fold cross-validation
- Data augmentation
- Adaptive threshold search
- Comprehensive logging and visualization

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your data in `data/train.csv` and `data/valid.csv`

## Configuration
Modify `config.yaml` to adjust experiment parameters.

## Running the Experiment
```bash
python main.py [--config config.yaml]
```

## Results
The model will generate:
- Learning curves
- Confusion matrix
- Detailed performance metrics
- Experiment logs

## Citation
(Add citation details for the SMM4H 2025 challenge)
```