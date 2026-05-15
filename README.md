# Garbage Classification with Transfer Learning

Deep Learning mini-project for university course (deadline: June 3, 2026).

Classification of household waste into 10 categories using CNN and Transfer Learning.

## Problem Statement

Task: Image classification (영상 데이터 분류).

Input: RGB images of waste items, resized to 224x224.

Output: One of 10 categories — metal, glass, biological, paper, battery, trash, cardboard, shoes, clothes, plastic.

## Dataset

Primary: [Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) by Suman Kunwar.
Approximately 13,000 labeled images across 10 classes.

Reference paper: [The Garbage Dataset (GD), 2026](https://arxiv.org/pdf/2602.10500).

Secondary: [TrashNet](https://github.com/garythung/trashnet), used as out-of-distribution test set to evaluate generalization.

## Project Structure

```
trash-classification/
├── notebooks/        Jupyter notebooks: EDA, training, experiments
├── src/              Reusable Python modules
├── experiments/      Results, plots, training logs
├── report/           Final PDF report
└── requirements.txt
```

## How to Run

Local development on Mac M1:

```
conda create -n trash python=3.11 -y
conda activate trash
pip install -r requirements.txt
jupyter notebook
```

Training on Kaggle:
Open any notebook from `notebooks/` on Kaggle, attach the dataset
`sumn2u/garbage-classification-v2`, and run all cells.

## Experiments

See `experiments/results.csv` for full hyperparameter tuning results.

| # | Model | Accuracy | Notes |
|---|-------|----------|-------|
| 1 | Baseline CNN | TBD | From scratch, 4 conv layers |
| 2 | ResNet50 (frozen) | TBD | Transfer Learning, classifier only |
| 3 | ResNet50 (fine-tuned) | TBD | Unfrozen last blocks |

## Commit Convention

This project follows Conventional Commits with extensions for ML work.

| Type | Purpose |
|------|---------|
| feat | New functionality |
| fix | Bug fix |
| exp | Experiment run, includes metrics in commit body |
| data | Dataset or preprocessing changes |
| report | Changes to the written report |
| refactor | Code restructuring without behavior change |
| perf | Performance optimization |
| docs | Documentation |
| chore | Maintenance, dependencies, configuration |
| style | Code formatting |

Example experiment commit:

```
exp(train): ResNet50 frozen base, Adam lr=1e-3 -> 89.4% val acc

- model: ResNet50, frozen base
- optimizer: Adam, lr=1e-3
- batch_size: 32
- epochs: 10
- augmentation: horizontal flip, rotation 15
- val_accuracy: 89.4%
- val_loss: 0.31
- training time: 23 min on Kaggle P100
```

## Report

See `report/report.pdf`.

## Author

University Deep Learning course mini-project, 2026.