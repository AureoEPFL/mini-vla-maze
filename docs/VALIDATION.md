# Validation record

## Executed experiments

The reference checkpoint was trained locally on CPU with Python 3.12.0 and PyTorch 2.14.1+cpu: 320 accepted demonstrations, 5,449 observations, 1,800 updates, checkpoint selected at update 1,000. The measured training time, including periodic validation, was 222 seconds. The original continuous-action baseline was retained as a separate experiment.

The 11×11 evaluation contains 656 records. The new 25×25 evaluation contains 36 records: hybrid and oracle on six accepted scenes in each of three families. All 18 hybrid large-map episodes succeeded. The neural action head is not evaluated at this size.

An independent Google Colab **CPU** runtime executed the portable notebook with Python 3.13.16 and PyTorch 2.11.0+cpu. All 692 outcomes, action counts, pushes, waits, collision counts and other non-timing episode fields match the local records. See [comparison](../results/colab/comparison.json), [platform provenance](../results/colab/provenance.json) and [execution capture](../results/colab/execution.png). CUDA was not tested.

Colab records were copied from an explicit JSON export displayed by the notebook after the browser download did not produce a local file. The checked-in notebook is the executable input notebook; the executed copy remains in Google Drive. The payload captures the source used for that run, including the original renderer docstring. Subsequent changes to that docstring and tests do not change inference or dynamics.

## Completed checks

- Nine unit tests: simulator contact rules, clock interlock, planner replay, language parsing, absence of a target marker, unsolvable cases, large-maze replay, tile vocabulary and spatially scalable visual heads.
- Default maze notebook executed from top to bottom, including two instructions on one image, animation export and result inspection. All three notebooks' Python cells parsed successfully.
- Training, validation and main-test seed sets are disjoint; held-out blue-disk commands are absent from training and validation targets.
- Aggregate main-benchmark success counts recounted from episode CSVs; local and Colab records compared field by field, excluding timings.
- Local Markdown links checked for missing targets. Scientific figures and representative animation frames visually inspected.
- Gallery selection code exercised for all eight cases with a minimal DOM fixture; referenced GIFs exist and decode. This is a script test, not browser rendering validation. The integrated browser could not reach the local preview server, and its security policy rejected a data-URL preview. Browser layout remains a manual visual check when opening `gallery.html`.
- Final ZIP extracted into a separate directory; every file checked against `SHA256SUMS.txt`; packaged-checkpoint demo and large-maze tests executed from that extracted copy.

## Renderer detail

The large renderer retains the same tile colours and interior pixels. Raster drawing order clips a one-pixel hazard-stroke spill at some tile boundaries. The tests check static scenes byte for byte and dynamic tile interiors explicitly. The local and Colab large-map runs use the same renderer; no visual-equivalence claim is made for every boundary pixel.

To rerun the tests: `python -m unittest discover -s tests -v` from the installed project. Timing and retrained weights may vary across hardware and software stacks. Intervals describe scene variability conditional on one trained checkpoint.
