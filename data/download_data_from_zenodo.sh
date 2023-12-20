#!/bin/bash

# Download and unzip
curl https://zenodo.org/records/10384998/files/stochastic-birefringence-data.zip --output "stochastic-birefringence-data.zip"
unzip stochastic-birefringence-data.zip

# Move input data to ../input/
mv matlab_orfs.dat ../input/
mv o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json ../input/

# Remove original zip files
rm stochastic-birefringence-data.zip
