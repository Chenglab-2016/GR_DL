# GR_DL

This repository contains model implementations and suggested workflows for the paper **“Glucocorticoid receptor activation reorganizes Wnt/LEF1 regulatory circuitry associated with therapeutic response in B-cell acute lymphoblastic leukemia.”**

The code prepares genomic datasets, trains sequence-based or motif-based deep learning models, and evaluates predicted genomic signal profiles. The supplied configurations cover the B-cell acute lymphoblastic leukemia cell lines 697, SUPB15, Nalm6, and RS411, using combinations of LEF1, glucocorticoid receptor (GR), and ATAC-seq signals.

## File introduction

| File or directory | Description |
| --- | --- |
| [`model_motif.py`](model_motif.py) | PyTorch implementation of an Enformer-style convolutional and transformer model. Supports raw DNA sequence inputs, motif representations initialized from MEME position weight matrices, optional additional signal tracks, and multiple prediction heads. |
| [`train_eval.py`](train_eval.py) | Training and testing entry point. Loads NPZ datasets, trains models, saves the best validation checkpoint, and exports test predictions and metrics. |
| [`dataProcessing.py`](dataProcessing.py) | Builds one NPZ file per genomic window from a reference FASTA, peak regions, input bigWigs, and target bigWigs. Supports chromosome-based splits and blacklist filtering. |
| [`dataProcessing_accel.py`](dataProcessing_accel.py) | Accelerated dataset preparation with multiprocessing, sharded NPZ output, and optional batched bigWig reads. |
| [`bash_for_sub_job.ipynb`](bash_for_sub_job.ipynb) | Example Bash commands for preprocessing, training, and testing on an LSF cluster using `bsub`. Adapt paths, environment names, queues, and resource requests to your system. |
| [`tsv/`](tsv/) | Input and target configuration files for different cell lines, signal combinations, and peak subsets. |
| [`LICENSE`](LICENSE) | MIT license. |

### Configuration files

The `cells*.tsv` files specify peak regions and input signals. All include `cell` and `narrowpeak` columns; remaining columns contain input bigWig paths.

| Configuration | Inputs or region selection |
| --- | --- |
| `cells.tsv` / `cells2.tsv` | DMSO LEF1 plus GR / GR alone. |
| `cells_atac.tsv` / `cells_atac2.tsv` | Adds DMSO ATAC signal to the corresponding input combination. |
| `cells_dexatac.tsv` / `cells_dexatac2.tsv` | Uses DEX ATAC signal with the corresponding input combination. |
| `cells_up.tsv`, `cells_down.tsv` | DMSO LEF1 and GR inputs at upregulated or downregulated GR/LEF1 regions. |
| `cells2_up.tsv`, `cells2_down.tsv` | GR inputs at the corresponding upregulated or downregulated regions. |
| `cells_dexup.tsv`, `cells_dexdown.tsv` | DMSO LEF1, GR, and DEX ATAC inputs at those region subsets. |
| `cells2_dexup.tsv`, `cells2_dexdown.tsv` | GR and DEX ATAC inputs at those region subsets. |
| `tracks.tsv` | One DEX LEF1 target track per cell line. |
| `tracks2.tsv` | Two LEF1 target tracks per cell line, ordered DEX then DMSO. |

Target configuration files use `cell`, `name`, and `path` columns. Target channels follow the row order for each cell; prediction heads must match that order and total channel count.

## Requirements and input data

The scripts require Python, PyTorch, NumPy, pyBigWig, and pyfaidx. Install these in your analysis environment:

```bash
python -m pip install torch numpy pyBigWig pyfaidx
```

For GPU training, use a PyTorch installation compatible with your CUDA environment. The cluster examples use a GPU and Bash; Jupyter is optional for viewing the notebook.

Provide the following files separately:

- A reference genome FASTA matching the signal tracks and peak coordinates (the notebook uses hg38).
- Peak regions in narrowPeak or BED format, plus input and target bigWig files referenced by the TSV configurations.
- An optional blacklist BED for the same genome assembly.
- A MEME-format motif file for motif-prior models, such as the `meme/consensus_pwms.meme` path used in the notebook.

These genomic inputs, motif files, generated datasets, and trained checkpoints are not included in this checkout. Edit the TSV paths before running. Relative paths inside TSV files are resolved from the working directory, not from the TSV directory.

## Suggested workflow

The following Bash examples run from the repository root. They use `cells_atac.tsv` and `tracks.tsv` to prepare all configured cell lines, then train and test a single-target RS411 model. Paths to external data are examples that must be replaced. These commands are starting configurations; reproducing a particular paper analysis requires its corresponding data, regions, and model settings.

### 1. Prepare datasets

```bash
python dataProcessing_accel.py \
  --fasta ./hg38/hg38.fa \
  --cells tsv/cells_atac.tsv \
  --tracks tsv/tracks.tsv \
  --out_dir datasets_atac \
  --sequence_length 2048 --target_length 1024 --bin_size 2 \
  --val_chroms chr8 chr14 \
  --test_chroms chr10 chr22 chr4 \
  --blacklist ./hg38/hg38-blacklist.v2.bed \
  --num_workers 4 --shard_size 2048 \
  --batch_bw_per_chrom --max_batch_bp 5000000 \
  --min_peak_distance_bp 1024
```

Omit `--blacklist` if no blacklist is supplied. Each cell directory contains `train_files.txt`, `val_files.txt`, `test_files.txt`, and an `npz_shards/` directory. NPZ files store one-hot DNA, additional per-base input signals, binned targets, and genomic metadata. Keep the same working directory when consuming lists containing relative paths.

For the simpler preprocessing implementation, use `dataProcessing.py` and omit `--num_workers`, `--shard_size`, `--batch_bw_per_chrom`, and `--max_batch_bp`. Its per-window files are stored in `npz/`.

### 2. Train a motif-based model

```bash
python train_eval.py \
  --mode train \
  --out_dir runs/RS411_motif \
  --train_list datasets_atac/RS411/train_files.txt \
  --val_list datasets_atac/RS411/val_files.txt \
  --heads "rs411=1" \
  --model_type motif-based-model \
  --motif_use_prior \
  --motif_pwm_path ./meme/consensus_pwms.meme \
  --sequence_length 2048 --target_length 1024 \
  --auto_fit_geometry \
  --batch 4 --epochs 30 --lr 1e-5 \
  --amp
```

The best validation checkpoint is saved to `runs/RS411_motif/best.pt`. To train the raw-sequence model, set `--model_type raw-sequence_model`, remove both motif-prior arguments, and choose a separate output directory. Use `--no_addition` for DNA-only inputs.

For two-target prediction, prepare data with `tracks2.tsv` and a matching input configuration such as `cells_atac2.tsv`, then use `--heads "dex=1,dmso=1"` for both training and testing.

### 3. Evaluate the saved model

```bash
python train_eval.py \
  --mode test \
  --out_dir runs/RS411_motif \
  --test_list datasets_atac/RS411/test_files.txt \
  --ckpt runs/RS411_motif/best.pt \
  --heads "rs411=1" \
  --batch 4
```

Testing reloads the model configuration from the checkpoint. Keep its motif file accessible at the recorded path. Outputs include `test_metrics.json` with per-head mean squared error and Pearson correlation, and `test_dump.npz` with predictions, targets, and metadata.

### Run suggestions

- Keep training, validation, and test chromosomes disjoint. Select settings using validation data and reserve the test split for final evaluation.
- Match sequence and target lengths between preparation and training. Sequence length must be divisible by bin size; target length cannot exceed the resulting number of bins. `--auto_fit_geometry` adjusts model pooling to retain enough output bins.
- Start with a small dataset using preprocessing option `--limit_per_split`, and reduce batch size if GPU memory is limited. Dataset loading also requires sufficient host memory.
- Keep input channel order and target head order consistent across splits. If using `--log1p_targets`, supply it during both training and testing; metrics then refer to transformed targets.
- Use the notebook as a cluster submission reference, updating its TSV paths to the files under `tsv/`. Allocate CPU resources consistent with the preprocessing worker count.
- Inspect available options with `python dataProcessing_accel.py --help` and `python train_eval.py --help`.

## Citation and license

When using this code, please cite the associated paper: **“Glucocorticoid receptor activation reorganizes Wnt/LEF1 regulatory circuitry associated with therapeutic response in B-cell acute lymphoblastic leukemia.”**

The code is distributed under the [MIT license](LICENSE).
