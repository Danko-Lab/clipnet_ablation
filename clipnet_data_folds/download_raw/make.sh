#!/bin/bash

# Download files from ftp_location.txt:
# ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/
# ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/

n=$(nproc --all)

echo snakemake --printshellcmds --reason --jobs "$((3 * n / 4))" --resources load=100
snakemake --printshellcmds --reason --jobs "$((3 * n / 4))" --resources load=100