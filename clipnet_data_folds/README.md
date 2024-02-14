# These are the instructions for downloading and processing GSE110638 data.

## Use conda to install environment from ../snakemake.yml

```bash
conda env create -f ../snakemake.yml
conda activate snakemake
```

## Install bwtool: https://github.com/CRG-Barcelona/bwtool
There may be some issues with installing this. This [issue](https://github.com/CRG-Barcelona/bwtool/issues/49) may help resolve things.

## Download raw data using the download_raw pipeline.

```bash
cd ./download_raw
bash make.sh
cd ../
```

## Process data types in order:

## Get procap peaks

```bash
cd ./get_peaks
bash make.sh
cd ../
```

## Get windows

```bash
cd ./get_window
bash make.sh
cd ../
```

## Process PRO-cap

```bash
cd ./procap
bash make.sh
cd ../
```

## Process DNase

```bash
cd ./dnase
bash make.sh
cd ../
```

## Get consensus sequences. Note that this will generate a large amount of temp .fna files until the next step can be completed

```bash
cd ./get_consensus_sequence
bash make.sh
cd ../
```

## Process sequence

```bash
cd ./sequence
bash make.sh
cd ../
```
