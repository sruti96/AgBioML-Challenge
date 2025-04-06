# flock-ag

## Getting started

First, download and install miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init
```

Then, create a conda environment with the necessary packages:

```bash
conda create -y -n flock-ag python=3.12
conda activate flock-ag

pip install -r requirements.txt
```




