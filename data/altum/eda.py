# Basic EDA so we understand the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load from feather files
betas = pd.read_feather("betas.arrow")
metadata = pd.read_feather("metadata.arrow")
probes = pd.read_csv("probes.txt", sep="\t")







