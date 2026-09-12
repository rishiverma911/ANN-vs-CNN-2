# Ready to Submit

## Upload this file

`notebooks/ann_vs_cnn_experiment.ipynb` (includes all code, outputs, and markdown)

## Checklist

- [x] Intel dataset ~25,000 images, 6 classes
- [x] 150×150 resize + normalization
- [x] ANN (dense) + CNN (conv, pool, dropout)
- [x] Full 14,034 train images (stratified val split)
- [x] Full 3,000 test images evaluated (all 6 classes)
- [x] Capacity-matched models (~10.8M vs ~10.7M params)
- [x] Training time, validation accuracy, parameters compared
- [x] CNN vs ANN markdown explanation + CUDA/TensorRT bonus
- [x] Notebook executed with visible outputs

## Results summary

| Model | Val accuracy | Test accuracy |
|-------|-------------|---------------|
| ANN   | 41.7%       | 41.6%         |
| CNN   | 85.4%       | 84.7%         |
