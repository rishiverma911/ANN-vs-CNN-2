# Intel Image Classification Dataset

Place the extracted Kaggle dataset here:

```text
data/raw/
├── seg_train/seg_train/{buildings, forest, glacier, mountain, sea, street}/
├── seg_test/seg_test/{buildings, forest, glacier, mountain, sea, street}/
└── seg_pred/seg_pred/
```

**~25,000 total images** across six natural scene categories (~14K train, ~3K test, ~7K unlabeled prediction images).

Download and extract:

```powershell
python download_dataset.py --force-extract
```
