#!/bin/bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yaml
conda activate gammaae 
pip install ipykernel 
python -m ipykernel install --user --name gammaae --display-name gammaae
