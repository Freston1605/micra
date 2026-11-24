# Dataset split tool

The labelled YOLO dataset for this project is **not** versioned in this
repository. It lives under the gitignored `data/` directory and is shared
through the project owner’s storage (institutional/author download).

`make_splits.py` turns a local copy of the dataset into a deterministic
`train` / `val` / `test` split. It:

1. verifies every image has a corresponding label (and vice-versa),
   plus the exact `names:` mapping in `dataset.yaml`,
2. mechanically moves files instead of copying them,
3. drops the two byte-identical duplicates (`LDPE 10`, `LDPE 12`),
4. rewrites `data/dataset.yaml` with `test: images/test`,
5. refuses to do anything if anything is missing or unexpected —
   and if the apply itself fails midway, moves everything back.

## Usage

```bash
.venv/bin/python tools/data/make_splits.py --root data
.venv/bin/python tools/data/make_splits.py --root data --apply
```

Without `--apply` the command is a dry run and changes nothing.

## Split design

Splits are **group-stratified by plate**, not randomly shuffled by image.
Images that came from the same sampling event or re-export are always kept
together:

- the three captures of `04-05-23 Sedcalci (4)` / `(5)` / `(6)`;
- the paired captures of `30-03-23 sed marino (6)` and `(7)`;
- all other images listed in `split_manifest.py`.

This is required because the same physical plate appears as multiple images.
A random split would leak information between training and validation and
would inflate the reported metrics.

The assignment was also optimised so the three splits contain roughly
proportional numbers of particles of each class while preserving plate groups.
It contains all three acquisition domains (spiked sediment, mixtures and pure
plates) in both val and test, so the metrics reflect the full distribution
rather than a single acquisition type.

`LDPE 11` is deliberately **not** in any split: it is kept in
`data/pending/` until a human annotates it.

## After the split

- `data/images/train`, `data/images/val`, `data/images/test`
- `data/labels/train`, `data/labels/val`, `data/labels/test`
- `data/dataset.yaml` — unchanged class names, plus the `test` split

Run `python -m pytest -q` across the repository after re-applying the split.

## Inventory (read-only)

`inventory_dataset.py` records what a copy of the dataset actually contains, so
a training execution can be traced back to the files that produced it. It
writes three tables and never renames, moves or edits an image, a label or the
split:

```bash
.venv/bin/python tools/data/inventory_dataset.py --root data \
  --manifest tools/data/split_manifest.py --output-dir data/inventory
```

- `images.csv` — one row per image: a rename-proof `image_id` (the first 16 hex
  of its SHA-256), relative image and label paths, split, plate group, size,
  label hash, annotation status and boxes per class.
- `plates.csv` — one row per plate group: sample type, spike status and the
  **source** of that claim. A category read from a file name is marked
  `filename`; `sample_id` stays empty until a laboratory record fills it.
- `dataset-version.json` — schema version, class names, per-split totals, the
  problems found and a `fingerprint` computed exactly as
  `polyx.pipeline.dataset` computes it, so it can be compared with the digest a
  training execution records.

Only the presence of EXIF metadata is recorded: capture dates travel because
they group acquisition sessions, GPS coordinates never do.

Exit status is 1 when the inventory found problems: a missing label, a plate
group crossing splits, byte-identical images in two folds, a declared split
that is not a directory, or no images at all. The tables are still written, so
the problems can be read from them and a healthy split still describes what it
holds beside a broken one. Status 2 is reserved for a usage or precondition
failure — no dataset declaration, unreadable class names, an existing inventory
without `--force` — where no reliable table can be produced. `--force`
overwrites an earlier inventory.

## Portable export

`export_renamed_dataset.py` writes a **copy** whose images and labels are named
from the image's SHA-256, so a download, a ZIP or a second filesystem cannot
mangle a name into a different meaning. The source tree is never touched.

```bash
.venv/bin/python tools/data/export_renamed_dataset.py --root data \
  --out data_portable --manifest tools/data/split_manifest.py
```

The copy keeps the same split, class map and bytes, and adds `mapping.csv`:
`image_id`, original image and label names, the new paths, the plate group and
source, both hashes and the box count. The split and class counts are verified
after the copy, so an export that loses a label fails instead of shipping.

Two images that shorten to the same 16-hex name are never merged. Each cause is
reported separately: a prefix collision between different images (widen the
prefix), one image carrying two different annotation files, a byte-identical
image in two folds, or a duplicate inside one fold. Byte-identical pixels with
different labels are refused rather than resolved, because one of the two
annotations is wrong and the export cannot tell which.

Status is 1 when the dataset cannot be exported safely (a missing or unpaired
label, a split that is not a directory, a duplicate image) and 2 for a usage or
precondition failure such as a missing declaration or an occupied output
folder.

Because the fingerprint is taken over the paths, a rename changes it
(`bc71bab5…` for the study split, `5d3b2473…` for its portable copy) while the
bytes and the mapping stay identical. Inventory the copy with
`--groups <export>/mapping.csv` to keep the plate groups the names no longer
carry:

```bash
.venv/bin/python tools/data/inventory_dataset.py --root data_portable \
  --groups data_portable/mapping.csv --output-dir data_portable/inventory
```


