#!/bin/bash

n=$(nproc --all)

echo snakemake --dryrun --printshellcmds --jobs "$((n / 2))" --resources load=100
snakemake --dryrun --printshellcmds --jobs "$((n / 2))" --resources load=100