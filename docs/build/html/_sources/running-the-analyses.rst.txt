Running the analyses
====================

Here, we detail how to rerun our code to recreate the data stored at https://doi.org/10.5281/zenodo.10384998.

Inference with fixed BBH merger rates
-------------------------------------

In Sect.~5.A of our paper, we obtain posteriors on birefringent parameters :math:`\kappa_D` and :math:`\kappa_z` with a few different fixed merger rates:

    1. A uniform-in-comoving volume merger rate
    2. A merger rate tracing star formation
    3. A merger rate tracing low-metallicity star formation, subject to a distribution of evolutionary time delays

1. Uniform-in-comoving volume merger rate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reproduce results using a uniform-in-comoving volume merger rate, do the following:

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ python run_birefringence_uniformRate.py

This script will call code defined in :file:`gridded_likelihood.py` to compute one-dimensional posteriors on :math:`\kappa_D` and :math:`\kappa_z`, as well as a joint posterior on both paramters.
Because this is only a two-dimensional space, we do this calculation via direct evaluation over grids of both parameters.

If on a computer cluster with slurm, you can instead queue a job to perform this calculation via:

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ sbatch launch_birefringence_uniformRate.sbatch

.. warning::

    The slurm batchfile noted above is set up to work on the KICP partition of Midway3 at the University of Chicago.
    You will need to edit it to use the appropriate queue and account on your own cluster!

The result will be a file containing one- and two-dimensional probability distributions over each birefringent parameter, created at the following location:

.. code-block:: bash

    data/fixed_rate_uniform.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_uniformRate.ipynb>`__

2. Merger rate tracing global star formation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reproduce results using a star-formation-tracing merger rate, do the following:

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ python run_birefringence_SFR.py

This script will call code defined in :file:`gridded_likelihood.py` to compute one-dimensional posteriors on :math:`\kappa_D` and :math:`\kappa_z`, as well as a joint posterior on both paramters.
Because this is only a two-dimensional space, we do this calculation via direct evaluation over grids of both parameters.

If on a computer cluster with slurm, you can instead queue a job to perform this calculation via:

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ sbatch launch_birefringence_SFR.sbatch

.. warning::

    The slurm batchfile noted above is set up to work on the KICP partition of Midway3 at the University of Chicago.
    You will need to edit it to use the appropriate queue and account on your own cluster!

The result will be a file containing one- and two-dimensional probability distributions over each birefringent parameter, created at the following location:

.. code-block:: bash

    data/fixed_rate_SFR.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_SFR.ipynb>`__

3. Merger rate tracing delayed low-metallicity star formation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To reproduce results using a merger rate following a delayed low-metallicity star formation rate, do the following:

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ python run_birefringence_delayedSFR.py

This script will call code defined in :file:`gridded_likelihood.py` to compute one-dimensional posteriors on :math:`\kappa_D` and :math:`\kappa_z`, as well as a joint posterior on both paramters.
Because this is only a two-dimensional space, we do this calculation via direct evaluation over grids of both parameters.

If on a computer cluster with slurm, you can instead queue a job to perform this calculation via:

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ sbatch launch_birefringence_delayedSFR.sbatch

.. warning::

    The slurm batchfile noted above is set up to work on the KICP partition of Midway3 at the University of Chicago.
    You will need to edit it to use the appropriate queue and account on your own cluster!

The result will be a file containing one- and two-dimensional probability distributions over each birefringent parameter, created at the following location:

.. code-block:: bash

    data/fixed_rate_delayedSFR.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR.ipynb>`__

Inference with variable BBH merger rate
---------------------------------------

In Sect.~5.B, we then infer birefringent parameters while simultaneously inferring and marginalizing over the redshift-dependent merger rate of BBHs.
We strongly recommend doing this analysis using a GPU on a computing cluster.
To do so, do the following:

.. code-block:: bash

    $ conda activate stochastic-birefringence-cuda
    $ cd code/
    $ sbatch launch_birefringence_variable_evolution.sbatch

.. warning::

    The slurm batchfile noted above is set up to work on the KICP partition of Midway3 at the University of Chicago.
    You will need to edit it to use the appropriate queue and account on your own cluster!

It is not strictly necessary to use a GPU/cluster; instead you can perform the analysis on your local machine with

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ sbatch run_birefringence_variable_evolution.py

Running locally, however, will likely take a significantly longer amount of time (perhaps several hours), rather than several minutes with a GPU.

However you choose to run this code, the result will be the following file:

.. code-block:: bash

    data/birefringence_variable_evolution.cdf

This file will contain the raw sampling chains from :code:`numpyro` as well as miscellaneous diagnostics.
To make this output a bit easier to work with for downstream applications, we will perform some post-processing and repackaging of samples into an :code:`hdf` format.
To do so, run

.. code-block:: bash

    $ cd data/
    $ python process_birefringence_variable_evolution.py

This will finally create the following file:

.. code-block:: bash

    data/birefringence_variable_evolution.hdf

Inference with individual baselines
-----------------------------------

In an Appendix, we investigate the per-baseline results to better understand the elevated "spike" appearing in some of our birefringent posteriors.
These per-baseline results are obtained by running the following scripts:

.. code-block:: bash

    code/run_birefringence_delayedSFR_HLO1.py
    code/run_birefringence_delayedSFR_HLO2.py
    code/run_birefringence_delayedSFR_HLO3.py
    code/run_birefringence_delayedSFR_HVO3.py
    code/run_birefringence_delayedSFR_LVO3.py

Running each of these proceeds completely analogously with the directions above, e.g.

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ python run_birefringence_delayedSFR_HLO1.py

As above, these can also be executed over slurm on a computing cluster via e.g.

.. code-block:: bash

    $ conda activate stochastic-birefringence
    $ cd code/
    $ sbatch launch_birefringence_delayedSFR_HLO1.sbatch

Running each of these scripts will produce the following output files:

.. code-block:: bash

    data/fixed_rate_delayedSFR_HLO1.hdf
    data/fixed_rate_delayedSFR_HLO2.hdf
    data/fixed_rate_delayedSFR_HLO3.hdf
    data/fixed_rate_delayedSFR_HVO3.hdf
    data/fixed_rate_delayedSFR_LVO3.hdf

.. note::

    Notebooks that demonstrates how to load in, inspect, and manipulate this output file can be found at

    * `inspect_birefringence_delayedSFR_HLO1 <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR_HLO1.ipynb>`__
    * `inspect_birefringence_delayedSFR_HLO2 <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR_HLO2.ipynb>`__
    * `inspect_birefringence_delayedSFR_HLO3 <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR_HLO3.ipynb>`__
    * `inspect_birefringence_delayedSFR_HVO3 <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR_HVO3.ipynb>`__
    * `inspect_birefringence_delayedSFR_LVO3 <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR_LVO3.ipynb>`__
