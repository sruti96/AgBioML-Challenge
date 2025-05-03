# Altum Dataset

This directory contains the data for the AltumAge dataset.

## How data was gathered

The data is available from this google drive: https://drive.google.com/drive/folders/1THmHwEpO4CwjNsJ-C1pXxPRghV8tU09a

The data was downloaded and then uploaded to the data directory. This yields two files:

- `betas_orig.arrow` - Feather file containing the beta values for each DNAm sample.
- `metadata_orig.arrow` - Feather format file contining the metadata for each DNAm sample.

Then, the data was split to create a held-out test set agents are not allowed to see:

```bash
python split_data.py
```

This created the following files:

- `meta_heldout.arrow` - Feather file containing the metadata for the held-out test set.
- `betas_heldout.arrow` - Feather file containing the beta values for the held-out test set.
- `metadata.arrow` - Feather file containing the metadata for data provided to agents.
- `betas.arrow` - Feather file containing the beta values for data provided to agents.
