# Final WRN test evaluation

Training job: 415005. Evaluation job: 418123.

Checkpoint: `/common/home/sjg256/cs462-cifar10/outputs/deadline_wrn/final/wrn28_10_aa_cosine_full/final_model.pt`

SHA-256: `b6b45598e9cd8fc6c322cbc223680a06ff1245d5d9ea614772978e4bc5a98a2d`

WRN-28-10, 36,479,194 parameters, dropout 0.3. Fresh training on all 50,000 images for 196 epochs, selected using 97.28% holdout accuracy.

Test images: **10,000**. Correct: **9719**. Cross-entropy loss: **0.0903988168**. Overall accuracy: **97.19%**.

Single unaugmented forward pass; ToTensor then saved Normalize; float32; no TTA. Strict state loading, model.eval(), torch.no_grad(), batch size 128, 79 batches, no dropped samples.

Every model parameter and buffer matched the checkpoint after evaluation. All pre-existing model, ZIP, and dataset hashes were verified unchanged.

| Class | Correct / total | Accuracy |
|---|---:|---:|
| airplane | 976 / 1000 | 97.60% |
| automobile | 988 / 1000 | 98.80% |
| bird | 968 / 1000 | 96.80% |
| cat | 929 / 1000 | 92.90% |
| deer | 981 / 1000 | 98.10% |
| dog | 956 / 1000 | 95.60% |
| frog | 986 / 1000 | 98.60% |
| horse | 982 / 1000 | 98.20% |
| ship | 980 / 1000 | 98.00% |
| truck | 973 / 1000 | 97.30% |

Confusion matrix: rows are true classes; columns are predicted classes.

| True / predicted | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| airplane | 976 | 0 | 5 | 2 | 1 | 0 | 1 | 0 | 14 | 1 |
| automobile | 1 | 988 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| bird | 2 | 0 | 968 | 6 | 10 | 8 | 4 | 0 | 2 | 0 |
| cat | 1 | 1 | 7 | 929 | 5 | 48 | 4 | 3 | 2 | 0 |
| deer | 0 | 0 | 4 | 6 | 981 | 3 | 3 | 3 | 0 | 0 |
| dog | 0 | 1 | 4 | 30 | 6 | 956 | 0 | 3 | 0 | 0 |
| frog | 3 | 0 | 4 | 5 | 1 | 0 | 986 | 1 | 0 | 0 |
| horse | 1 | 0 | 2 | 2 | 3 | 10 | 0 | 982 | 0 | 0 |
| ship | 8 | 6 | 3 | 0 | 0 | 0 | 0 | 1 | 980 | 2 |
| truck | 2 | 20 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 973 |

Normalization was loaded unchanged from the final checkpoint and computed on the 50,000 training images:

```json
{
  "mean": [
    0.4913996786151962,
    0.4821584083946079,
    0.4465309144454656
  ],
  "std": [
    0.24703223246328174,
    0.2434851280000557,
    0.26158784172796457
  ],
  "source_count": 50000
}
```

Previously evaluated legacy model scored 90.83%; this is the single authorized evaluation of the fixed WRN final model. No post-test selection or tuning.

The expanded WRN recipe uses residual connections, Kaiming initialization, momentum/Nesterov, cosine scheduling, AutoAugment and Cutout. The historical reports describe the original course restrictions and legacy model; their scores do not apply to this WRN.
