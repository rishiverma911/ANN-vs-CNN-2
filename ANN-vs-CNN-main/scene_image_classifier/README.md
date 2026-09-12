# Scene Image Classifier: ANN vs CNN

PyTorch experiment comparing a baseline **ANN (MLP)** with a **CNN** on the **Intel Image Classification** dataset (~25,000 images, six scene classes).

## Assignment alignment

| Requirement | Implementation |
|-------------|----------------|
| Intel dataset (~25K, 6 classes) | Full Kaggle download verified (~24,335 raw images) |
| 150×150 resize + normalization | `get_transforms()` in `project_utils/data.py` |
| ANN baseline (dense layers) | `ANNClassifier` with comparable ~10M parameters |
| CNN (conv, pooling, dropout) | `CNNClassifier` |
| Compare training time, val accuracy, parameters | Saved to `results/metrics/model_comparison.csv` |
| Jupyter notebook + CNN explanation | `notebooks/ann_vs_cnn_experiment.ipynb` |

## Dataset layout

```text
data/raw/
├── seg_train/seg_train/{buildings, forest, glacier, mountain, sea, street}/  (~14,034)
├── seg_test/seg_test/{buildings, forest, glacier, mountain, sea, street}/     (~3,000)
└── seg_pred/seg_pred/                                                          (~7,301 unlabeled)
```

**Supervised training** uses every labeled image in `seg_train` with a seeded stratified train/validation split. **Evaluation** uses the complete official `seg_test` split (all 3,000 images, all six classes).

## Quick start

```powershell
cd scene_image_classifier
python download_dataset.py --force-extract   # if dataset not extracted yet
python run_full_experiment.py                # train + evaluate end-to-end
python run_full_experiment.py --eval-only    # re-evaluate saved checkpoints
jupyter notebook notebooks/ann_vs_cnn_experiment.ipynb
```

## Outputs

| Artifact | Path |
|----------|------|
| Best models | `results/models/best_ann_model.pth`, `best_cnn_model.pth` |
| Comparison CSV | `results/metrics/model_comparison.csv` |
| Confusion matrices | `results/figures/ann_confusion_matrix.png`, `cnn_confusion_matrix.png` |
| Training curves | `results/figures/ann_training_curves.png`, `cnn_training_curves.png` |

## Reproducibility

`RANDOM_SEED = 42` for Python, NumPy, PyTorch, and the stratified train/validation split.

## License

Academic / educational use.
