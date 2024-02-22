#!/bin/bash

n=$(nproc --all)

echo snakemake --printshellcmds --jobs "$((n / 2))" --resources load=100
snakemake --printshellcmds --jobs "$((n / 2))" --resources load=100