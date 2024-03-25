VAE-Based Molecular Simulation
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/vae-mol-sim/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/vae-mol-sim/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/VAE-Based Molecular Simulation/branch/main/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/VAE-Based Molecular Simulation/branch/main)


A library of Tensorflow code facilitating the use of VAEs in molecular simulations


### Installation Instructions

#### To obtain the code:

1) From the command line...

    Move to a directory where you want the folder containing the repository of code to be installed, then type:
        `git clone https://git.uark.edu/jm217/vae-mol-sim.git`
    That will create a folder called 'vae-mol-sim' with all of the files from the repository.

2) From a web browser...

    In the main repository page, click on "Code" and select a compressed file format for the code download. Once downloaded, decompress the folder to a location of your choosing.

#### To install the code...

Navigate to the 'vae-mol-sim' directory in a command line. If you are in the environment where you want the package installed, simply type
    `pip install ./`
Note that will install the code and any dependencies into whatever environment is currently activated.

If you need to create an environment, the recommendation is to use conda. An example is provided in 'devtools/conda-envs/basic_example_env.yaml'. To create a conda environment while in a command line in the 'vae_mol_sim' directory, run
    `conda env create -f devtools/conda-envs/basic_example_env.yaml`
To activate it, run
    `conda activate vae-mol-sim`
You can then install the package into this environment with
    `pip install ./ --no-deps`
The last flag '--no-deps' prevents unnecessary re-installation of dependencies already covered in the conda environment.

A brief note on dependencies... Note that, currently, there are difficulties with the latest version of tensorflow_probability (0.24) being incoompatible with the latest version of keras used by tensorflow version 2.16.1. Details can be found [here](https://github.com/tensorflow/probability/releases). This would be fine if tensorflow_probability correctly included that dependency (for tf_keras, an earlier keras version packaged with a different name), but it does not. Until this is figured out, I have pinned dependencies to tensorflow 2.15.0 or earlier and tensorflow_probability 0.23.0 or earlier. This also requires that python be no more than version 3.11 (only tensorflow 2.16.1 is compatible with python version 3.12). If you are feeling brave and know what you are doing, please feel free to go ahead and install everything manually with the latest versions, run the tests (with `pytest` from the vae-mol-sim directory) and report back what happens.

#### To test out some examples...

You will want to be familiar with Jupyter notebooks. In brief, navigate to the 'examples' folder and, from a command line, run
    `jupyter notebook`
That should open a web browser with all files in the directory shown. You can open one and interactively run code, including changing it to see what happens.

#### tldr;
```
git clone https://git.uark.edu/jm217/vae-mol-sim.git
cd vae_mol_sim
conda env create -f devtools/conda-envs/basic_example_env.yaml
conda activate vae-mol-sim
pip install ./ --no-deps
```

### Building the Documentation
Right now, there is no fancy website hosting the documentation, however, the documentation can be built and viewed locally in an appropriate browser. To do this, follow the instructions in the 'README.md' file inside the 'docs' directory, which will involve installing the packages 'sphinx' and 'sphinx_rtd_theme' (those do not need to be installed in the same environment in which you have installed the vaemolsim package, but they are included in basic_example_env.yaml).

### Copyright

Copyright (c) 2023, Jacob I. Monroe


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
