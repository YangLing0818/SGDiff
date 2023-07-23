## Step 1: Install COCO API
To train new models you will need to install the [COCO Python API](https://github.com/cocodataset/cocoapi). Unfortunately installing this package via pip often leads to build errors, but you can install it from source like this:

```bash
cd ~
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
python setup.py install
```

## Step 2: Preparing the data

### Visual Genome
Run the following script to download and unpack the relevant parts of the Visual Genome dataset:

```bash
bash scripts/download_vg.sh
```

This will create the directory `datasets/vg` and will download about 15 GB of data to this directory; after unpacking it will take about 30 GB of disk space.

After downloading the Visual Genome dataset, we need to preprocess it. This will split the data into train / val / test splits, consolidate all scene graphs into HDF5 files, and apply several heuristics to clean the data. In particular we ignore images that are too small, and only consider object and attribute categories that appear some number of times in the training set; we also igmore objects that are too small, and set minimum and maximum values on the number of objects and relationships that appear per image.

```bash
python scripts/preprocess_vg.py
```

This will create files `train.h5`, `val.h5`, `test.h5`, and `vocab.json` in the directory `datasets/vg`.

### COCO
Run the following script to download and unpack the relevant parts of the COCO dataset:

```bash
bash scripts/download_coco.sh
```

This will create the directory `datasets/coco` and will download about 21 GB of data to this directory; after unpacking it will take about 60 GB of disk space.
