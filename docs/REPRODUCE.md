# Reproduce the experiments

Run from the project folder after `python -m pip install -e .`.

## Train and evaluate

```bash
python -m mini_vla_maze.train --out runs/maze --episodes 320 --steps 1800 --seed 17
python -m mini_vla_maze.evaluate --checkpoint runs/maze/weights.pt --out runs/maze/eval --episodes 16
python -m mini_vla_maze.visualize --run runs/maze --out runs/maze/media
python -m mini_vla_maze.large --checkpoint runs/maze/weights.pt --out runs/large --episodes 6
```

Training uses CUDA when available; both evaluators load the checkpoint on CPU. Use a clean output directory when changing data generation. The measured local training run took 222 seconds, including periodic validation. Runtime varies by hardware and software.

## Rebuild publication assets

```bash
python -m mini_vla_maze.report_assets --root .
```

This reads the supplied `results/maze` and `results/large` records and writes the architecture figure (PNG/SVG/PDF), illustrated benchmark, labelled animations, scene images and HTML viewer. It does not invent or overwrite benchmark scores. The episode selections come from saved records.

## Colab

Upload `notebooks/03_portable_validation.ipynb` and run all cells. Its compressed payload includes the source needed for the experiment and the original checkpoint. It evaluates rather than retrains. The last cell downloads a result archive. Keep provenance and episode CSVs separate from local results.

The checked-in Colab run used Python 3.13.16 / PyTorch 2.11.0+cpu. The local reference used Python 3.12.0 / PyTorch 2.14.1+cpu. Compare outcomes and action counts, not wall-clock timings. The notebook was executed in Colab; the text records were archived from an explicit visible JSON export after browser downloading did not return a local file.

## Integrity

`SHA256SUMS.txt` records final file hashes. Unit tests cover dynamics and the perception interface. Acceptance/rejection manifests record the solvable subset used for reporting. Retraining may produce different weights on another software stack.
