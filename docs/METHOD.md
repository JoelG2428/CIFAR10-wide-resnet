# Training and reproduction

The four files in `deadline_wrn/` preserve the actual training implementation. Training defaults to disabled and requires a Slurm GPU allocation. Historical deadline controllers, machine-specific shell scripts and duplicate checkpoints are omitted from the showcase.

Place the existing CIFAR-10 Python batches under `data/cifar-10-batches-py/`. Downloads are disabled. `outputs/split_indices.npz` preserves the original 45,000/5,000 split. Run from this repository root in a compatible GPU environment:

```bash
python deadline_wrn/run.py
# Inside an allocated Slurm GPU job, with the dataset prepared:
python deadline_wrn/run.py --stage train --name reproduction_holdout --execute
python deadline_wrn/run.py --stage final --name reproduction_full --select-run reproduction_holdout --execute
```

The holdout stage runs 200 epochs; final training selects the best validation epoch from that new run, with the earliest epoch winning ties. A new run can select a different epoch or achieve a different score. For the historical experiment, the selected epoch was 196 and best validation accuracy was 97.28%. New outputs go under `outputs/deadline_wrn/` and are ignored by Git. Use unique names; existing run directories are protected. Same-run recovery accepts `--resume` with saved optimizer, scheduler and RNG states.

The official test evaluation is documented in `results/FINAL_REPORT.md` and its numerical records. The training entry point has no official-test stage; regenerating figures never repeats that evaluation. The original checkpoint is separate from new reproduction outputs.

The seed, split, configuration, normalization and numerical histories are preserved. GPU kernels, framework versions and hardware can affect reproducibility. The recorded versions describe the original machine; `requirements.txt` does not freeze that machine's full environment.

## Checkpoint integrity

The original checkpoint SHA-256 is `b6b45598e9cd8fc6c322cbc223680a06ff1245d5d9ea614772978e4bc5a98a2d`. Strict loading uses the exact architecture and sets evaluation mode. The checkpoint includes training metadata and state, which makes it larger than a weights-only file. Keep it outside ordinary Git commits and provide it separately when publishing.
