Running the analyses
====================

Here, we detail how to rerun our code to recreate the data stored at https://doi.org/10.5281/zenodo.10384998.

Inference of birefringence parameters with fixed BBH merger rates
-----------------------------------------------------------------

In Sect.~5 of our paper, we obtain posteriors on birefringent parameters :math:`\kappa_D` and :math:`\kappa_z` with a few different fixed merger rates:

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

The result will be a file containing one- and two-dimensional probability distributionsn over each birefringent parameter, created at the following location:

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

The result will be a file containing one- and two-dimensional probability distributionsn over each birefringent parameter, created at the following location:

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

The result will be a file containing one- and two-dimensional probability distributionsn over each birefringent parameter, created at the following location:

.. code-block:: bash

    data/fixed_rate_delayedSFR.hdf

.. note::

    A notebook that demonstrates how to load in, inspect, and manipulate this output file can be found `here <https://github.com/tcallister/stochastic-birefringence/blob/main/data/inspect_birefringence_delayedSFR.ipynb>`__
