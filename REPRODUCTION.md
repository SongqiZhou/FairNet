# Reproducing the FairNet paper

This document is the operational companion to the NeurIPS 2025 paper
*FairNet: Dynamic Fairness Correction without Performance Loss via Contrastive
Conditional LoRA*. It covers how to obtain the data, which command produces
which table, and — just as importantly — every place where the released code
makes a choice the paper left open or states differently.

- Paper: [proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/81f2d59479a96afd8056db9468254515-Abstract-Conference.html)
  · [PDF](https://papers.nips.cc/paper_files/paper/2025/file/81f2d59479a96afd8056db9468254515-Paper-Conference.pdf)
  · [arXiv](https://arxiv.org/abs/2510.19421)

## 1. Environment

```bash
python -m pip install -e ".[test,experiments]"
```

PyTorch must match the local CUDA build; install it from
[pytorch.org](https://pytorch.org/get-started/locally/) first if a GPU is used.
Every result file records the exact package versions, GPU model, and git commit
it was produced with, under the `environment` key.

## 2. Data

### CelebA

Supplementary C.1 uses the aligned images with the standard partition. The
authors' original download is a rate-limited Google Drive link, so the helper
script rebuilds the identical layout from a public mirror:

```bash
python scripts/prepare_celeba.py --out data/celeba
```

This writes `img_align_celeba/` (202,599 JPEGs at the original 178x218),
`list_attr_celeba.txt`, and `list_eval_partition.txt` with the official
162,770 / 19,867 / 19,962 split. The script verifies every parquet shard and
re-fetches truncated downloads, and it fails loudly rather than producing a
partial dataset.

The task predicts **Male** (attribute 20) with **Blond Hair** (attribute 9) as
the sensitive attribute. The resulting cells are extremely skewed, which is what
makes the benchmark a fairness benchmark:

| Split | not blond, not male | not blond, male | blond, not male | blond, male |
| --- | ---: | ---: | ---: | ---: |
| train (162,770) | 71,629 | 66,874 | 22,880 | **1,387** |
| validation (19,867) | 8,535 | 8,276 | 2,874 | **182** |
| test (19,962) | 9,767 | 7,535 | 2,480 | **180** |

**Read WGA with its error bar.** The worst group is blond-and-male, and it has
only **180 test images**. One flipped prediction moves WGA by 0.56 points, and
the binomial standard error around 80% accuracy is about 3 points. Single-seed
CelebA WGA differences under roughly 3 points are therefore noise; run several
seeds and compare means. The aggregator reports mean +- standard deviation
whenever more than one seed is present.

### MultiNLI and HateXplain

Both are fetched on first use by `fairnet.create_multinli_loaders` and
`fairnet.create_hatexplain_loaders`; no manual download step is needed. MultiNLI
comes from the `nyu-mll/multi_nli` parquet release. HateXplain is read from the
authors' own `dataset.json` and `post_id_divisions.json` rather than the Hugging
Face mirror, because that mirror is a loading script and `datasets` 5 refuses to
execute scripts; the resulting splits are the canonical 15,383 / 1,922 / 1,924.

## 3. Running

One configuration, one seed:

```bash
python experiments/run_experiment.py \
  --config celeba_full --seed 0 --data-root data/celeba
```

The whole CelebA suite (Table 1, Table 3, Supplementary Tables 5, C, and I)
across three seeds:

```bash
bash experiments/run_celeba_suite.sh data/celeba results 0 1 2
```

Results land in `results/<config>/seed<k>.json`. Render the comparison tables
with:

```bash
python experiments/aggregate_results.py --results results
```

The aggregator diffs each run against the paper's published numbers, which are
transcribed in `experiments/paper_reference.py`, and marks whether the
reproduction is at least as good (higher ACC and WGA, lower EOD).

### Configurations

| Config | Paper row |
| --- | --- |
| `celeba_erm` | Table 1, ERM |
| `celeba_full` | Table 1, FairNet-Full |
| `celeba_partial` | Table 1, FairNet-Partial |
| `celeba_unlabeled` | Table 1, FairNet-Unlabel |
| `celeba_ablate_detector` | Table 3, w/o detector |
| `celeba_ablate_contrastive` | Table 3, w/o contrastive loss |
| `celeba_ablate_both` | Table 3, w/o both |
| `celeba_full_ablate_contrastive` | Supplementary Table C |
| `celeba_partial --set labeled_fraction=...` | Supplementary Table 5 |
| `--sweep-threshold` on any variant | Supplementary Table I |
| `multinli_{erm,full,partial,unlabeled}` | Table 1, MultiNLI columns |
| `hatexplain_{bert,distilbert}[_race]` | Table 2 |

### Shared Stage 1

`--stage1-checkpoint PATH` trains the ERM base model once and reuses it for
every variant at that seed. This is both cheaper and cleaner: the Table 1 rows
then differ only in Stages 2-4, so any change in WGA is attributable to the
correction mechanism rather than to a different base model.

## 4. Two things that decide whether you match Table 1

Most of the pipeline is mechanical. Two settings are not, and both are easy to
get wrong in a way that silently turns FairNet into a no-op.

### 4.1 The Stage 1 budget

FairNet exists to repair **minority underfitting**, so how converged the Stage 1
base model is decides how much there is left to repair. This is not a minor
hyperparameter:

* Stop Stage 1 early and the base model still underfits the minority group. The
  contrastive hinge is open for most anchors, the LoRA modules receive a real
  gradient, and the correction has something to do.
* Train Stage 1 to convergence and the minority representations already sit
  close to their same-class majority prototype. The gap `d_neg - d_pos` grows,
  the triplet hinge is satisfied almost everywhere, and the correction quietly
  does nothing - on a converged CelebA backbone the hinge opens on only a few
  percent of batches.

The second regime produces a *better* ERM baseline in overall accuracy, so it is
not wrong, it is just a different question. `celeba_base.yaml` uses the short
budget, which is the regime Table 1's ERM row sits in.
`celeba_erm_converged.yaml` and `celeba_full_converged.yaml` run the same
pipeline on a fully converged backbone for comparison, since that is the more
realistic deployment case.

Stage 4 records `stage4_contrastive_active_rate` in every result file and warns
when the hinge never opens, so you can tell which regime you are in.

### 4.2 The contrastive margin is not transferable

The margin is compared against `d_neg - d_pos`, whose scale is set by how
converged the backbone is. A margin tuned on one Stage 1 budget will not be
right for another: too small and the hinge never opens, too large and it opens
for the common class of the minority group as well, diluting the update the rare
class needs.

Retune it whenever you change the backbone or the Stage 1 budget, on validation
WGA, as Supplementary C.3.2 prescribes. `--set contrastive_margin=...` makes
that a one-line sweep.

## 5. Deviations, and why

The paper leaves several choices open or, in two places, states something that
its own reported numbers contradict. Each is resolved here explicitly rather
than silently.

### 5.1 Worst-group accuracy is over label-by-group cells

Supplementary C.4 defines
`WGA = min(P(Y_hat = Y | S = 0), P(Y_hat = Y | S = 1))`, a minimum over the two
sensitive groups. The numbers in Table 1 follow the other convention — the
minimum over `(task label, sensitive group)` cells — and two checks make this
unambiguous:

* On CelebA, the blond group is 29,983 images of which only 1,749 are male.
  Predicting "not male" for every blond image already scores 94.2% on that
  group, so the reported ERM WGA of **77.9%** cannot be a minimum over sensitive
  groups. It is consistent with the accuracy on the small blond-and-male cell.
* On MultiNLI, the reported ERM pair (ACC 82.6, WGA 67.3) matches the standard
  six-cell worst-group accuracy for this dataset in the GroupDRO line of work,
  which is also the convention every baseline in Table 1 was published under.

`compute_fairness_metrics` therefore reports the label-by-group minimum as
`worst_group_accuracy` by default, and always returns both quantities under the
explicit names `worst_label_group_cell_accuracy` and
`worst_sensitive_group_accuracy`. Passing `wga_definition="sensitive_group"`
(or setting `FairNetConfig.wga_definition`) restores the literal C.4 formula.

### 5.2 The contrastive distance, and the only scale where the paper's margin works

Equation 2 is a triplet hinge `[D(z_a, z_p) - D(z_a, z_n) + margin]_+`. The main
paper writes the distance as "e.g., squared Euclidean", Supplementary C.3.2 says
"Euclidean distance was used", and C.3.2 tunes the margin "between 0.1 and 1.0".
Only one combination makes all of that true at once, and it is worth spelling
out because the wrong choice silently disables the entire method.

What matters is the scale of the gap `d_neg - d_pos` over minority anchors,
because the hinge is open exactly when that gap falls below the margin.

Raw squared Euclidean distance cannot work with the stated margin. CelebA CLS
representations and their prototypes have norms in the tens, so the distances
land in the hundreds, and a margin three orders of magnitude smaller can never
open the hinge. Instrumenting Stage 4 on that configuration shows the
contrastive term pinned at exactly zero for every batch: the LoRA matrices
receive no fairness signal at all, and FairNet reduces to a no-op.

L2-normalising the representations fixes the scale but is still not enough with
the squared form, whose gap sits well above the top of the paper's range.
Taking C.3.2 literally — *plain* Euclidean distance over L2-normalised
representations — puts distances in [0, 2] and the gap in the region where a
margin in [0.1, 1.0] genuinely selects between "correct only the hardest
anchors" and "correct roughly half of them". That is the default here:
`contrastive_distance="euclidean"`, `contrastive_normalize=True`.
`"squared_euclidean"` and `"cosine"` remain available.

Stage 4 records `stage4_contrastive_active_rate` in every result file and prints
a warning if the hinge never opens, so a future scale mismatch shows up instead
of quietly producing a method that does nothing.

### 5.3 The contrastive anchors must be class-balanced

Each sensitive attribute has exactly one shared `(A, B)` LoRA pair, and Stage 4
only ever sees the triggered samples, i.e. the minority group. On CelebA that
group is 22,880 blond women against 1,387 blond men. Averaged unweighted, the
common class dominates the update, and the single shared correction learns to
push *every* triggered face toward it - including the blond men it is supposed
to rescue. The measurable effect is that the blond-and-male cell ends up well
*below* the ERM baseline the method exists to improve, while the already-strong
blond-and-female cell edges up.

`class_balanced_contrastive=True` weights each anchor by the inverse frequency
of its task class within the batch, which restores the behaviour the paper
describes: accuracy preserved, worst group up, EOD down.

The same reasoning applies to Equation 3's `L_task`. Restricted to triggered
samples — which is what it reduces to, since a closed gate contributes no LoRA
gradient — an unweighted task loss reinforces exactly the imbalance the method
is meant to repair. `stage4_task_weight` therefore defaults to `0.0`, matching
Section 3.4's description of Stage 4 as "fine-tuning of LoRA modules using a
contrastive loss formulation"; raising it carries Equation 3's task term, and
`class_balanced_stage4_task` keeps it from re-introducing the skew.

### 5.4 Validation sensitive labels

Section 5.1 states that FairNet-Partial has no sensitive labels on the
validation set and FairNet-Unlabeled has none anywhere. Selecting checkpoints on
validation WGA would leak exactly the labels those settings withhold, so:

* **Stage 1** is plain ERM in every setting and selects on validation accuracy.
  This also keeps the ERM baseline honest — selecting the base model on WGA
  would quietly make the baseline fairness-aware.
* **Stage 4** selects on validation WGA only for FairNet-Full, which
  Supplementary C.3.1 grants validation sensitive labels. Partial and Unlabeled
  select on validation accuracy.

WGA is still *logged* for all settings, from the test and validation splits, for
reporting only. This is the conservative reading: it can only make the
reproduction harder, never easier.

### 5.5 Hyperparameters the paper gives as ranges

Supplementary C states ranges rather than values for the activation threshold
(0.5-0.8, grid searched), the contrastive margin (0.1-1.0, tuned), the detector
MLP depth ("1-2 hidden layers"), and the detector placement ("intermediate
layers ... can be flexibly chosen"). It gives no learning rates, epoch counts,
optimiser, or schedule. `experiments/configs/*.yaml` pins concrete values, each
annotated with the paper clause it comes from. The architecture itself is not a
guess: the ViT of Supplementary C.2 with 8 layers, 8 heads, intermediate size
768, 64x64 inputs, and 16x16 patches has 29.57M parameters, matching
Supplementary Table F exactly.

Batch size 128 is likewise pinned down by the paper: Supplementary Table 5
reports 162,688 labelled training samples at 100% coverage, and
`162770 // 128 * 128 = 162688`, so the original runs used batch size 128 with
the final partial batch dropped.

### 5.6 MultiNLI splits

Supplementary C.1 says "standard train, validation-matched, and test-matched
splits", but MultiNLI's test-matched labels are withheld by the benchmark and
are not publicly available. `split_scheme: "matched"` (default) trains on the
official train split and deterministically halves validation-matched into
validation and test. `split_scheme: "groupdro"` instead reproduces the 50/20/30
random split of Sagawa et al., under which several Table 1 baselines were
originally published.

The negation cue list defaults to the paper's own examples (`not`, `n't`,
`never`); `GROUPDRO_NEGATION_WORDS` provides the Gururangan/Sagawa list
(`nobody`, `no`, `never`, `nothing`) for direct comparison with that line of
work.

### 5.7 HateXplain sensitive attributes

Section 5.3 targets the "African American" and "Female" demographics. The
released annotations mark target communities per annotator, so a post is
assigned to a group when a majority of its annotators marked the corresponding
community (`African` for race, `Women` for gender). Table 2's progressive
strategy — race first, then gender — maps onto the `*_race` configs and the
two-attribute configs respectively.

## 6. Compute

The paper used a single A100 80GB (Supplementary C.6). The CelebA experiments
here were run on an NVIDIA L40S. The CelebA backbone is small — 17 tokens per
image at 64x64 with 16x16 patches — so one Stage 1 run is on the order of ten
minutes and a full variant adds a few minutes on top.
