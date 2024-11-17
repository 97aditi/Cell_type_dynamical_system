# Cell Type Dynamical Systems 

This is the code repository for cell-type specific linear dynamical systems, based on [Jha, Gupta, Brody, Pillow (NeurIPS 2024)](https://www.biorxiv.org/content/10.1101/2024.07.08.602520v1).


#### Setup instructions:
1. Clone the repository with submodules.
``` 
    git clone --recurse-submodules https://github.com/yourusername/your-repo-name.git
    cd Cell_type_dynamical_system
```
2. Run the setup script:
``` 
    bash setup.sh
```
3.  Follow the prompts to set up your MOSEK license.
You're now ready to use the project!

#### Demo script
fit_ctds_on_simulated_data.ipynb describes how to create and fit a CTDS model to simulated data.