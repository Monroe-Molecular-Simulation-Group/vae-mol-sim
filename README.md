VAE-Based Molecular Simulation
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/vae-mol-sim/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/vae-mol-sim/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/VAE-Based Molecular Simulation/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/VAE-Based Molecular Simulation/branch/main)


A library of Tensorflow code facilitating the use of VAEs in molecular simulations


### Installation Instructions

To obtain the code:
1) From the command line...
    Move to a directory where you want the folder containing the repository of code to be installed, then type:
        `git clone https://git.uark.edu/jm217/vae-mol-sim.git`
    That will create a folder called 'vae-mol-sim' with all of the files from the repository.
2) From a web browser...
    In the main repository page, click on "Code" and select a compressed file format for the code download. Once downloaded, decompress the folder to a location of your choosing.

Next, navigate to the 'vae-mol-sim' directory in a command line. If you are in the environment where you want the package installed, simply type
    `pip install ./'
Note that will install the code and any dependencies into whatever environment is currently activated.

If you need to create an environment, the recommendation is to use conda. An example is provided in 'devtools/conda-envs/example_env.yaml'. That includes molecular simulation packages, so if you want something more barebones, use 'devtools/conda-envs/test_env.yaml'. To create a conda environment while in a command line in the 'vae_mol_sim' directory, run
    `conda env create -f devtools/conda-envs/example_env.yaml`
To activate it, run
    `conda activate vae-mol-sim`
You can then install the package into this environment with
    `pip install ./ --no-deps`
The last flag '--no-deps' prevents unnecessary re-installation of dependencies already covered in the conda environment.

To test out some examples, you will want to be familiar with Jupyter notebooks. In brief, navigate to the 'examples' folder and, from a command line, run
    `jupyter notebook`
That should open a web browser with all files in the directory shown. You can open one and interactively run code, including changing it to see what happens.

tldr;
```
git clone https://git.uark.edu/jm217/vae-mol-sim.git
cd vae_mol_sim
conda env create -f devtools/conda-envs/example_env.yaml
conda activate vae-mol-sim
pip install ./ --no-deps
```

### Copyright

Copyright (c) 2023, Jacob I. Monroe


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
