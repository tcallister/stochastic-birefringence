Getting started
===============

Setting up your environment
----------------------------

To make it as easy as possible to reproduce our results and/or figures, the `environment.yml` file can be used to build a conda environment containing the packages needed to run the code in this repository.
To set up this environment, do the following:

**Step 0**. Make sure you have conda installed. If not, see e.g. https://docs.conda.io/en/latest/miniconda.html

**Step 1**. Do the following:

.. code-block:: bash

    $ conda env create -f environment.yml

This will create a new conda environment named *stochastic-birefringence*

**Step 2**. To activate the new environment, do

.. code-block:: bash

    $ conda activate autoregressive-bbh-inference 

You can deactivate the environment using :code:`conda deactivate`

.. note::

    We provide an additional file `environment-cuda.yml` that can be used as above to build an environment with cuda-enabled `jax` and `numpyro` for use with GPUs.
    This is particularly helpful when simultaneously fitting for birefringent coefficients as well as the black hole merger rate using `run_birefringence_variable_evolution.py`.
    Following the above steps, but substituting this environment file, will create another environment called *stochastic-birefringence-cuda*.


Downloading input files and inference results
---------------------------------------------

Datafiles containing the output of our inference codes are hosted on Zenodo.
All data needed to regenerate figures and/or rerun our analyses can be found at https://doi.org/10.5281/zenodo.10384998.
To download this input/output data locally, you can do the following:

.. code-block:: bash

    $ cd data/
    $ . download_data_from_zenodo.sh

This script will populate the :code:`data/` directory with datafiles containing processed outputs of our analyses.
These output files can be inspected by running the jupyter notebooks also appearing in the :code:`data/` directory.
The script will also place several files in the :code:`input/` directory, which are needed to rerun analyses and/or regenerate figures.
