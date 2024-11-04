# Cell Type Dynamical Systems 

This is the code repository for cell-type specific linear dynamical systems, based on [Jha, Gupta, Brody, Pillow (NeurIPS 2024)](https://www.biorxiv.org/content/10.1101/2024.07.08.602520v1).


#### Setup instructions:
This is built on top of the [SSM package](https://github.com/lindermanlab/ssm/tree/master/ssm). 

```
conda env create --file=ctds.yml
conda activate ctds
cd ssm
pip install -e .
```

We also use MOSEK to solve constrained optimization, under the hood. This requires access to a license file that is free for academics. Please follow the instructions [here](https://docs.mosek.com/10.2/licensing/quickstart.html#i-don-t-have-a-license-file-yet) to download the lic file and save it in your home directory as described (/Users/myusername/mosek/mosek.lic for OSX).


#### Demo script
fit_ctds_on_simulated_data.ipynb describes how to create and fit a CTDS model to simulated data.