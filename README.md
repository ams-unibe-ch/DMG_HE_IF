# DMG_HE_IF — predicting spatial protein markers from H&E in pediatric DMG

<!-- After the first Zenodo release, replace XXXXXXX with the record number of the concept DOI -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)

Code for the AI-based digital pathology analysis in

> Laternser S.\*, Brüningk S.\*, De Micheli A. J.\*, McDonough E.\*, *et al.*, Nazarian J.
> **Mapping the Tumor Microenvironment of Pediatric Diffuse Midline Glioma using a Spatial Proteomic Atlas with AI-Based Application.**
>  \*shared first authorship
> Primary authors: J. Bugajska; S. Brüningk

The package installs a commandline tool, `dmg-he`, that runs every step of the analysis.
From a tissue microarray (TMA) stained by multiplexed immunofluorescence (MxIF), the
pipeline learns to predict, for 22 markers, whether a **100 µm × 100 µm tile** of the
(virtual) H&E image contains marker-positive cells. Tiles are embedded with a pretrained
pathology foundation model and one lightGBM classifier is trained per marker. One TMA block
is used for training with patient-level 5-fold cross-validation, the second block serves as
the cross-block test set.

```
cell-by-cell table ─┐
thresholds per core ├─► 1 prepare ─► 2 tile ─► 3 split ─► 4 extract-features ─► 5 collect-features ─► 6 train ─► 7 evaluate
virtual H&E slide  ─┘   (cores)      (tiles,    (5 CV        (foundation model)     (one table           (lightGBM   (tables, plots,
                                      labels)    folds)                              per block)           per marker)  spatial maps)
```

## Installation

Python ≥ 3.10. In a fresh virtual environment:

```bash
git clone https://github.com/ams-unibe-ch/DMG_HE_IF.git
cd DMG_HE_IF
pip install -e .              # core pipeline (tiling, lightGBM, evaluation)
pip install -e ".[torch]"     # + RetCCL, CTransPath, UNI, Virchow, Prov-GigaPath
pip install -e ".[remedis]"   # + REMEDIS (TensorFlow)
```

Feature-extractor specifics:

| `--model` | Model | Features | Weights |
|---|---|---|---|
| `uni` | UNI, ViT-L/16 ([Chen et al., Nat Med 2024](https://github.com/mahmoodlab/UNI)) | 1024 | Hugging Face `MahmoodLab/uni` (gated) |
| `virchow` | Virchow, ViT-H/14, class + mean patch token ([Vorontsov et al., Nat Med 2024](https://huggingface.co/paige-ai/Virchow)) | 2560 | Hugging Face `paige-ai/Virchow` (gated) |
| `provgigapath` | Prov-GigaPath tile encoder ([Xu et al., Nature 2024](https://github.com/prov-gigapath/prov-gigapath)) | 1536 | Hugging Face `prov-gigapath/prov-gigapath` (gated) |
| `ctranspath` | CTransPath, Swin-T ([Wang et al., Med Image Anal 2022](https://github.com/Xiyue-Wang/TransPath)) | 768 | `ctranspath.pth` from the authors |
| `retccl` | RetCCL, ResNet-50 ([Wang et al., Med Image Anal 2023](https://github.com/Xiyue-Wang/RetCCL)) | 2048 | `best_ckpt.pth` from the authors |
| `remedis` | REMEDIS ([Azizi et al., Nat Biomed Eng 2023](https://physionet.org/content/medical-ai-research-foundations/)) | 4096 | TF Hub SavedModel (PhysioNet) |
| `rgb_stats` | colour statistics — fast baseline used by the tests | 15 | – |

* Gated Hugging Face models: request access on the model page, then `export HF_TOKEN=hf_...`.
* CTransPath and RetCCL: clone the authors' repository and add it to `PYTHONPATH`
  (CTransPath additionally needs the modified `timm` shipped with TransPath, so install it in
  a separate environment). Set the weight paths in the config.

## Input data

The imaging and patient data are not part of this repository. Put the following files into one folder:

| File | Content |
|---|---|
| cell-by-cell table, one per block | QuPath export: one row per cell with `TMA core`, `Centroid X µm`, `Centroid Y µm` and `<Cell\|Nucleus>: <MARKER> mean` intensity columns |
| threshold table, one per block | Positivity threshold per core (rows) and marker (columns); a row called `Name` holds the marker names |
| virtual H&E slide, one per block | RGB whole-slide TIFF, 0.325 µm/pixel, in the same coordinate frame as the cell centroids |
| sample metadata | One row per core and block with `Name` (core), `TMA` (block id), `PID` (patient), `Diagnosis`, `Dissemination` |

## Quick start

```bash
dmg-he init-config config.yaml        # writes all settings with comments
# edit config.yaml: data_dir, work_dir, file names (and weight paths if needed)

dmg-he prepare           -c config.yaml
dmg-he tile              -c config.yaml
dmg-he split             -c config.yaml
dmg-he extract-features  -c config.yaml -m uni
dmg-he collect-features  -c config.yaml -m uni
dmg-he train             -c config.yaml -m uni
dmg-he evaluate          -c config.yaml -m uni
dmg-he plot-core         -c config.yaml -m uni --marker H3K27M --core M-22
dmg-he status            -c config.yaml           # what has been computed so far
```

Every command skips work that is already done (use `--overwrite` to redo it), so interrupted
runs can simply be restarted. `dmg-he <command> --help` lists all options, e.g.

```bash
dmg-he train -c config.yaml -m uni --marker H3K27M NESTIN --fold 0 1   # selected markers / folds
dmg-he train -c config.yaml -m uni --fold full     # one model on the whole training block
dmg-he evaluate -c config.yaml -m uni virchow provgigapath ctranspath retccl remedis   # comparison
```

### On a SLURM cluster

The scripts in `slurm/` run the steps as job arrays. Set your config and virtual environment
in `slurm/env.sh`, then submit everything with dependencies from the repository root:

```bash
bash slurm/submit_all.sh uni virchow provgigapath
```

or step by step (`sbatch slurm/02_tile.sh`, `sbatch slurm/04_extract_features.sh uni`, ...).
Work is split over array tasks with `--shard slurm`, so the array size can be chosen freely.

## Method summary and settings

All settings live in the config file (defaults in
[`src/dmg_he/default_config.yaml`](src/dmg_he/default_config.yaml)).

* **Cores** with fewer than 10 segmented cells are skipped. Cell centroids are converted to
  slide pixels (0.325 µm/px).
* **Tiles**: a 100 µm (307 px) window is shifted over each core's cell bounding box in steps
  of half its edge length (50 µm), horizontally then vertically. Windows with at least 5 cells
  are kept.
* **Labels**: a cell is positive for a marker if its mean intensity in the marker's compartment
  (nucleus for SOX2, KI67 and H3K27M; whole cell otherwise) exceeds the per-core threshold. A
  tile is positive if it contains at least one positive cell.
* **Cross-validation**: the cores of the training block are split into 5 folds with
  `StratifiedGroupKFold` — grouped by patient and stratified by DMG/non-DMG core.
* **Classifier**: per marker and fold, standard scaling (fit on the training tiles) followed by
  lightGBM (`class_weight="balanced"`). Learning rate, number of leaves, number of trees and
  L1/L2 regularisation are tuned by grid search with inner 3-fold CV on ROC AUC.
* **Evaluation**: ROC AUC, AUPRC and AUPRC relative to the positive-class prevalence
  (threshold-free), plus accuracy, balanced accuracy, precision, recall, specificity and F1 at
  a probability threshold of 0.5. Reported as mean ± SD over the 5 fold models, on the held-out
  fold (`val`) and the complete test block (`test`), and for test-block subgroups (DMG, DMG
  primary/metastasis, other primary tumours).

## Outputs

Everything is written to `work_dir`:

```
cores/<block>/metadata.csv                core metadata, thresholds, number of tiles
tiles/labels_<block>.csv                  one row per tile: position, n_cells, positive cells per marker
tiles/<block>/png/<core>/<tile>.png       tile images
splits.json, splits_summary.csv           cross-validation folds
features/<model>/features_<block>.pkl     tile embeddings
models/<model>/<marker>/fold<k>/          model.joblib, predictions_val.csv, predictions_test.csv, metrics.json
results/<model>/metrics_summary.csv       mean and SD over folds per marker and evaluation set
results/<model>/metrics_per_fold.csv      all metrics per marker, fold and evaluation set
results/<model>/barplot_<set>.png         ROC AUC and relative AUPRC per marker
results/<model>/spatial_maps/             true / predicted / difference maps (plot-core)
results/comparison/                       all models side by side (evaluate with several -m)
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The test builds a small synthetic TMA with the same file formats and runs the whole pipeline
with the `rgb_stats` extractor (no GPU or model downloads needed).

## Citation

If you use this code, please cite the article above (full reference to be added on
publication) and the archived software release on Zenodo (DOI to be added). GitHub's
**"Cite this repository"** button (from [`CITATION.cff`](CITATION.cff)) provides the
software citation in APA and BibTeX format.

## License

[MIT](LICENSE). The pretrained foundation models are subject to their own licenses and terms
of use.
