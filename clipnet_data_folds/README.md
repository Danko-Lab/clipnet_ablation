# These are the instructions for subsampling the data for the clipnet project

## Use conda to install environment from ../snakemake.yml

```bash
conda env create -f ../snakemake.yml
conda activate snakemake
```

```python
import pandas as pd
import random
pd.read_csv("../data_spec/nonempty_prefixes.txt")
ids=pd.read_csv("../data_spec/nonempty_prefixes.txt", header=None)[0].to_list()
for i in [5, 10, 20, 230]:
    random.shuffle(ids)
    pd.Series(ids).to_csv(f"subsample_prefixes_n{i}_run1.txt", index=False, header=None)
```
