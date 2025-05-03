# Basic EDA so we understand the data

import pandas as pd
import numpy as np

# Load from feather files
betas = pd.read_feather("betas_orig.arrow")
metadata = pd.read_feather("metadata_orig.arrow")

# Selected based on similar(ish) age-ranges and tissue types vs rest of studies
heldout_studies = [
    'E-GEOD-63347', 'E-GEOD-77955', 'E-GEOD-71955', 
    'E-GEOD-54399', 'E-GEOD-83334', 'E-GEOD-64511', 
    'E-GEOD-59457', 'GSE38608', 'E-GEOD-62867', 
    'GSE36642', 'GSE20242', 'E-GEOD-51954', 
    'E-MTAB-487', 'E-GEOD-56515', 'E-GEOD-73832', 
    'E-GEOD-59509', 'E-GEOD-72338', 'E-GEOD-61454', 
    'E-GEOD-30870', 'E-GEOD-71245'
]

# Create heldout dataset
metadata_heldout = metadata[metadata.dataset.isin(heldout_studies)]
betas_heldout = betas.loc[metadata_heldout.index]
assert np.all(metadata_heldout.index == betas_heldout.index)
metadata_heldout.to_feather('meta_heldout.arrow')
betas_heldout.to_feather('betas_heldout.arrow')

# Create agent dataset
metadata_agents = metadata[~metadata.dataset.isin(heldout_studies)]
betas_agents = betas.loc[metadata_agents.index]
assert np.all(metadata_agents.index == betas_agents.index)
metadata_agents.to_feather('metadata.arrow')
betas_agents.to_feather('betas.arrow')




